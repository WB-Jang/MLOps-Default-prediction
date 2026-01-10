"""Airflow DAG for data ingestion using MongoDB."""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.database.mongodb import mongodb_client
from src.data.data_loader import DataLoader
from loguru import logger
import pandas as pd


default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


def ingest_raw_data(**context):
    """Ingest raw data from CSV and store in MongoDB."""
    logger.info("Starting data ingestion task")
    
    # Load raw data from CSV file
    data_path = "./data/raw/synthetic_data.csv"
    logger.info(f"Loading raw data from {data_path}")
    
    mongodb_client.connect()
    
    try:
        loader = DataLoader(data_path)
        df = loader.load_raw_data()
        
        # Convert DataFrame to records
        records = df.to_dict('records')
        
        # Store data in batches to MongoDB
        batch_size = 100
        total_inserted = 0
        
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            
            # Insert batch into MongoDB raw_data collection
            result = mongodb_client.get_collection("raw_data").insert_many(batch)
            total_inserted += len(result.inserted_ids)
            
            logger.info(f"Inserted batch {i // batch_size} ({len(batch)} records) to MongoDB")
        
        # Store ingestion metadata
        mongodb_client.get_collection("ingestion_logs").insert_one({
            "timestamp": datetime.utcnow(),
            "data_source": "loan_applications",
            "file_path": data_path,
            "total_records": len(records),
            "status": "completed"
        })
        
        logger.info(f"Total {total_inserted} records stored in MongoDB raw_data collection")
        
        # Return metadata for next task
        return {
            "total_records": total_inserted,
            "data_path": data_path
        }
        
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        logger.error("Please run: python generate_raw_data.py")
        raise
    except Exception as e:
        logger.error(f"Error during data ingestion: {e}")
        raise
    finally:
        mongodb_client.disconnect()


def process_raw_data(**context):
    """Process raw data from MongoDB."""
    logger.info("Starting data processing task")
    
    # Get ingestion info from previous task
    ti = context['ti']
    ingestion_info = ti.xcom_pull(task_ids='ingest_raw_data')
    
    mongodb_client.connect()
    
    try:
        # Load raw data from MongoDB
        raw_data_cursor = mongodb_client.get_collection("raw_data").find()
        raw_data = list(raw_data_cursor)
        
        logger.info(f"Retrieved {len(raw_data)} records from MongoDB")
        
        # Convert to DataFrame for processing
        df = pd.DataFrame(raw_data)
        
        # Remove MongoDB _id field
        if '_id' in df.columns:
            df = df.drop(columns=['_id'])
        
        # Basic preprocessing: encode categorical features
        from src.data.data_loader import DataLoader
        loader = DataLoader(ingestion_info['data_path'])
        loader.df = df
        loader._detect_column_types()
        loader.encode_categorical_columns()
        
        # Store processed data back to MongoDB
        processed_records = loader.df.to_dict('records')
        
        # Clear existing processed data
        mongodb_client.get_collection("processed_data").delete_many({})
        
        # Insert processed data
        result = mongodb_client.get_collection("processed_data").insert_many(processed_records)
        
        logger.info(f"Stored {len(result.inserted_ids)} processed records in MongoDB")
        
        # Store processing metadata
        mongodb_client.get_collection("processing_logs").insert_one({
            "timestamp": datetime.utcnow(),
            "records_processed": len(processed_records),
            "status": "completed"
        })
        
        return {
            "processed_records": len(processed_records)
        }
        
    finally:
        mongodb_client.disconnect()


def mark_data_ready(**context):
    """Mark processed data as ready for training."""
    logger.info("Marking data as ready for training")
    
    # Get processing info from previous task
    ti = context['ti']
    processing_info = ti.xcom_pull(task_ids='process_raw_data')
    
    mongodb_client.connect()
    
    try:
        # Update ingestion status to mark data as ready
        mongodb_client.get_collection("data_status").insert_one({
            "timestamp": datetime.utcnow(),
            "status": "ready_for_training",
            "processed_records": processing_info['processed_records'],
            "collection": "processed_data"
        })
        
        logger.info("Data marked as ready for training in MongoDB")
        
    finally:
        mongodb_client.disconnect()


# Define DAG
with DAG(
    'data_ingestion_pipeline',
    default_args=default_args,
    description='Data ingestion pipeline using MongoDB',
    schedule_interval=timedelta(hours=1),
    catchup=False,
    tags=['mongodb', 'ingestion'],
) as dag:
    
    # Task 1: Ingest raw data to MongoDB
    ingest_task = PythonOperator(
        task_id='ingest_raw_data',
        python_callable=ingest_raw_data,
        provide_context=True,
    )
    
    # Task 2: Process raw data from MongoDB
    process_task = PythonOperator(
        task_id='process_raw_data',
        python_callable=process_raw_data,
        provide_context=True,
    )
    
    # Task 3: Mark data as ready for training
    mark_ready_task = PythonOperator(
        task_id='mark_data_ready',
        python_callable=mark_data_ready,
        provide_context=True,
    )
    
    # Define task dependencies
    ingest_task >> process_task >> mark_ready_task
