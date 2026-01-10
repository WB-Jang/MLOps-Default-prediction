"""Airflow DAG for data ingestion using Kafka."""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.kafka.producer import kafka_producer
from src.kafka.consumer import RawDataConsumer
from src.database.mongodb import mongodb_client
from src.data.data_loader import DataLoader
from src.data.preprocessing import prepare_data_for_kafka
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
    """Ingest raw data from CSV and send to Kafka."""
    logger.info("Starting data ingestion task")
    
    # Load raw data from CSV file
    data_path = "./data/raw/synthetic_data.csv"
    logger.info(f"Loading raw data from {data_path}")
    
    try:
        loader = DataLoader(data_path)
        df = loader.load_raw_data()
        
        # Connect to Kafka producer
        kafka_producer.connect()
        
        # Convert data to records format
        records = prepare_data_for_kafka(df)
        
        # Send data in batches to Kafka
        batch_size = 100
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            batch_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "data_source": "loan_applications",
                "batch_number": i // batch_size,
                "records": batch
            }
            
            # Send to Kafka raw_data topic
            kafka_producer.send_raw_data(batch_data, key=f"batch_{i // batch_size}")
            logger.info(f"Sent batch {i // batch_size} ({len(batch)} records) to Kafka")
        
        logger.info(f"Total {len(records)} records sent to Kafka raw_data topic")
        
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        logger.error("Please run: python generate_raw_data.py")
        raise
    except Exception as e:
        logger.error(f"Error during data ingestion: {e}")
        raise
    finally:
        kafka_producer.disconnect()


def process_raw_data(**context):
    """Process raw data from Kafka."""
    logger.info("Starting data processing task")
    
    # Connect to MongoDB
    mongodb_client.connect()
    
    # Create consumer for raw data
    consumer = RawDataConsumer(group_id="airflow_processor")
    consumer.connect()
    
    try:
        # Process messages
        def process_message(message):
            logger.info(f"Processing message: {message.get('data_source')}")
            # Add actual processing logic here
            
            # Store metadata in MongoDB if needed
            mongodb_client.get_collection("raw_data_logs").insert_one({
                "timestamp": datetime.utcnow(),
                "message": message,
                "status": "processed"
            })
        
        # Consume batch of messages
        messages = consumer.consume_batch(batch_size=100, timeout_ms=5000)
        for msg in messages:
            process_message(msg['value'])
        
        logger.info(f"Processed {len(messages)} messages")
        
    finally:
        consumer.disconnect()
        mongodb_client.disconnect()


def send_processed_data(**context):
    """Send processed data to Kafka."""
    logger.info("Sending processed data to Kafka")
    
    kafka_producer.connect()
    
    try:
        # Example processed data
        processed_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "processing_status": "completed",
            "records_processed": 0
        }
        
        kafka_producer.send_processed_data(processed_data, key="processed")
        logger.info("Processed data sent to Kafka")
        
    finally:
        kafka_producer.disconnect()


# Define DAG
with DAG(
    'data_ingestion_pipeline',
    default_args=default_args,
    description='Data ingestion pipeline using Kafka',
    schedule_interval=timedelta(hours=1),
    catchup=False,
    tags=['kafka', 'ingestion'],
) as dag:
    
    # Task 1: Ingest raw data
    ingest_task = PythonOperator(
        task_id='ingest_raw_data',
        python_callable=ingest_raw_data,
        provide_context=True,
    )
    
    # Task 2: Process raw data
    process_task = PythonOperator(
        task_id='process_raw_data',
        python_callable=process_raw_data,
        provide_context=True,
    )
    
    # Task 3: Send processed data
    send_task = PythonOperator(
        task_id='send_processed_data',
        python_callable=send_processed_data,
        provide_context=True,
    )
    
    # Define task dependencies
    ingest_task >> process_task >> send_task
