"""
Airflow DAG for data generation, storage, and processing using DataGenLoaderProcessor.
This DAG implements the daily data generation -> storage -> processing workflow.
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import sys
import os
from loguru import logger
import pandas as pd

# 프로젝트 루트 경로 추가 (모듈 import를 위해)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.database.mongodb import mongodb_client
from src.data.data_gen_loader_processor import DataGenLoaderProcessor

# 설정
DATA_PATH = "/opt/airflow/data/raw/synthetic_data.csv"
PROCESSED_DATA_PATH = "/opt/airflow/data/processed/processed_data.csv"
DATASET_NAME = "loan_default_prediction"
DISTRIBUTION_MODEL_PATH = "./src/data/distribution_model.pkl"

default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=3),
    'execution_timeout': timedelta(minutes=20),
}

def generate_data_task(**context):
    """
    DataGenLoaderProcessor를 사용하여 Distribution model로부터 합성 데이터를 생성합니다.
    """
    logger.info("Starting data generation using distribution model...")
    
    # 데이터 디렉토리 생성
    # os.makedirs removed - directory exists via docker volume
    
    try:
        processor = DataGenLoaderProcessor(data_path=DATA_PATH)
        # Use distribution model to generate synthetic data
        processor.data_generation(data_quantity=10000)
        
        logger.info(f"Data generated successfully at {DATA_PATH}")
        
        # Store generation metadata
        context['ti'].xcom_push(key='generation_timestamp', value=datetime.utcnow().isoformat())
        
    except FileNotFoundError as e:
        logger.warning(f"Distribution model not found: {e}")
        logger.info("Fallback: Using alternative data generation method")
        # Fallback to alternative generation if distribution model is missing
        import subprocess
        subprocess.run(["python", "generate_raw_data.py", "--num-rows", "10000"], check=True)

def upload_to_mongodb_task(**context):
    """
    생성된 Raw CSV 파일을 MongoDB GridFS에 업로드합니다.
    (upload_raw_data 사용)
    """
    logger.info("Starting upload to MongoDB...")
    
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found.")

    mongodb_client.connect()
    try:
        file_id = mongodb_client.upload_raw_data(
            file_path=DATA_PATH, 
            dataset_name=DATASET_NAME
        )
        logger.info(f"File uploaded to GridFS with ID: {file_id}")
        
        # XCom에 file_id 저장 (필요 시 다운스트림에서 사용)
        context['ti'].xcom_push(key='raw_data_file_id', value=file_id)
        
    finally:
        mongodb_client.disconnect()

def preprocess_data_task(**context):
    """
    DataGenLoaderProcessor를 사용하여 데이터를 로드하고 전처리를 수행합니다.
    전처리된 데이터를 MongoDB에 저장합니다.
    """
    logger.info("Starting data preprocessing...")
    
    # 처리된 데이터 디렉토리 생성
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    
    processor = DataGenLoaderProcessor(data_path=DATA_PATH)
    
    # 1. 데이터 로딩
    df = processor.load_raw_data()
    logger.info(f"Loaded data shape: {df.shape}")
    
    # 2. 결측치 처리 (IsNull_cols)
    df = processor.IsNull_cols()
    
    # 3. Object 타입 변환 (obj_cols)
    df = processor.obj_cols()
    
    # 4. 날짜 데이터 처리 (dt_data_handling)
    processor.dt_data_handling()
    
    # 5. 데이터 축소 (data_shrinkage)
    processor.data_shrinkage()
    
    # 6. 인코딩 및 스케일링 (detecting_type_encoding)
    scaled_df, str_col_list, num_col_list, nunique_str = processor.detecting_type_encoding()
    
    # 전처리된 데이터 확인
    logger.info("Preprocessing completed.")
    logger.info(f"Processed columns: {processor.df.columns.tolist()}")
    logger.info(f"Processed data shape: {scaled_df.shape}")
    
    # 7. 전처리된 데이터를 CSV로 저장
    processor.save_processed_data(PROCESSED_DATA_PATH)
    logger.info(f"Processed data saved to {PROCESSED_DATA_PATH}")
    
    # 8. MongoDB에 전처리된 데이터 저장
    # MongoDB에 전처리된 데이터 저장
    mongodb_client.connect()
    try:
        import numpy as np
        import json
        
        # DataFrame을 JSON으로 변환 후 다시 파싱 (완전한 직렬화)
        json_str = scaled_df. to_json(orient='records', date_format='iso')
        processed_data_records = json.loads(json_str)
        
        logger.info(f"Prepared {len(processed_data_records)} records for MongoDB")
        
        # Insert processed data into MongoDB
        collection = mongodb_client.get_collection("processed_data")
        
        # 기존 데이터 삭제
        collection. delete_many({"dataset_name": DATASET_NAME})
        logger.info("Cleared existing processed data")
        
        # 배치 저장
        batch_size = 500
        for i in range(0, len(processed_data_records), batch_size):
            batch = processed_data_records[i:i+batch_size]
            batch_with_metadata = [
                {
                    **record,
                    "processing_timestamp": datetime.utcnow().isoformat(),  # ISO 문자열로 변환
                    "dataset_name": DATASET_NAME
                }
                for record in batch
            ]
            collection.insert_many(batch_with_metadata)
            logger.info(f"✅ Inserted batch {i//batch_size + 1}: {len(batch)} records")
        
        logger.info(f"✅ Total {len(processed_data_records)} records inserted into MongoDB")
        
        # Store processing metadata
        metadata_collection = mongodb_client.get_collection("processing_metadata")
        
        # numpy 타입을 Python 네이티브 타입으로 변환
        cat_max_dict_clean = {str(k): int(v) for k, v in nunique_str.items()}
        
        metadata_collection.insert_one({
            "timestamp": datetime.utcnow(),
            "dataset_name": DATASET_NAME,
            "num_records": int(len(processed_data_records)),
            "num_categorical_features": int(len(str_col_list)),
            "num_numerical_features": int(len(num_col_list)),
            "categorical_features": str_col_list,
            "numerical_features": num_col_list,
            "cat_max_dict": cat_max_dict_clean,
            "status":   "completed"
        })
        
        logger.info("✅ Processing metadata stored in MongoDB")
        
    finally:
        mongodb_client.disconnect()
        logger.info("✅ MongoDB disconnected")
        
    logger.info("✅✅✅ Preprocessing completed successfully!")
    return {"status": "success", "records": len(processed_data_records)}
        
    

with DAG(
    'data_pipeline_dag',
    default_args=default_args,
    description='Generate, Upload (GridFS), and Preprocess Data',
    schedule_interval=timedelta(days=1),
    catchup=False,
    max_active_runs=1,  # ⭐ 추가:  동시 실행 제한
) as dag: 

    t1_generate = PythonOperator(
        task_id='generate_data',
        python_callable=generate_data_task,
    )

    t2_upload = PythonOperator(
        task_id='upload_raw_data',
        python_callable=upload_to_mongodb_task,
    )

    t3_preprocess = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data_task,
        execution_timeout=timedelta(minutes=15),  # ⭐ 추가
        retries=1,  # ⭐ 이 태스크는 1회만 재시도
    )

    t1_generate >> t2_upload >> t3_preprocess