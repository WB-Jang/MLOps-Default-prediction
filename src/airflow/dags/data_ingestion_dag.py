"""
Airflow DAG for data generation, storage, and processing using DataGenLoaderProcessor.
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import sys
import os
from loguru import logger

# 프로젝트 루트 경로 추가 (모듈 import를 위해)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.database.mongodb import mongodb_client
from src.data.data_gen_loader_processor import DataGenLoaderProcessor

# 설정
DATA_PATH = "../data/synthetic_data.csv"
DATASET_NAME = "loan_default_prediction"

default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def generate_data_task(**context):
    """
    DataGenLoaderProcessor를 사용하여 합성 데이터를 생성합니다.
    """
    logger.info("Starting data generation...")
    
    # 데이터 디렉토리 생성
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    
    processor = DataGenLoaderProcessor(data_path=DATA_PATH)
    processor.data_generation(data_quantity=10000)
    
    logger.info(f"Data generated successfully at {DATA_PATH}")

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
    """
    logger.info("Starting data preprocessing...")
    
    processor = DataGenLoaderProcessor(data_path=DATA_PATH)
    
    # 1. 데이터 로딩
    df = processor.load_raw_data()
    logger.info(f"Loaded data shape: {df.shape}")
    
    # 2. 결측치 처리 (IsNull_cols)
    df = processor.IsNull_cols()
    
    # 3. Object 타입 변환 (obj_cols)
    df = processor.obj_cols()
    
    # 4. 날짜 데이터 처리 (dt_data_handling)
    # dt_data_handling은 내부적으로 self.df를 수정하므로 반환값을 받지 않아도 반영될 수 있으나,
    # 코드 확인 결과 반환값이 없거나 self.df를 수정하는 구조라면 아래와 같이 호출합니다.
    processor.dt_data_handling()
    
    # 전처리된 데이터 확인
    logger.info("Preprocessing completed.")
    logger.info(f"Processed columns: {processor.df.columns.tolist()}")
    logger.info(f"Processed data sample:\n{processor.df.head()}")

    # (옵션) 전처리된 데이터를 다시 저장하거나 다음 단계로 넘길 수 있습니다.
    # processor.df.to_csv("./data/processed/processed_data.csv", index=False)

with DAG(
    'data_pipeline_dag',
    default_args=default_args,
    description='Generate, Upload (GridFS), and Preprocess Data',
    schedule_interval=timedelta(days=1),
    catchup=False,
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
    )

    # 태스크 순서 정의
    t1_generate >> t2_upload >> t3_preprocess
