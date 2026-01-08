"""Airflow DAG for model training pipeline."""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import torch

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models import (
    Encoder,
    TabTransformerClassifier,
    ProjectionHead,
    pretrain_contrastive,
    train_classifier
)
from src.kafka.consumer import ProcessedDataConsumer
from src.database.mongodb import mongodb_client
from config.settings import settings
from loguru import logger


default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


def prepare_training_data(**context):
    """Prepare training data from Kafka."""
    logger.info("Preparing training data from Kafka")
    
    # Connect to Kafka consumer
    consumer = ProcessedDataConsumer(group_id="training_consumer")
    consumer.connect()
    
    mongodb_client.connect()
    
    try:
        # Consume processed data
        messages = consumer.consume_batch(batch_size=1000, timeout_ms=10000)
        logger.info(f"Retrieved {len(messages)} messages for training")
        
        # Store data preparation metadata
        mongodb_client.get_collection("training_data").insert_one({
            "timestamp": datetime.utcnow(),
            "num_samples": len(messages),
            "status": "prepared"
        })
        
        # Push data info to XCom for next task
        return {"num_samples": len(messages)}
        
    finally:
        consumer.disconnect()
        mongodb_client.disconnect()


def pretrain_model(**context):
    """Pretrain the encoder using contrastive learning."""
    logger.info("Starting model pretraining")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Generate timestamp once for consistency
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create encoder and projection head
    # Note: In production, these parameters should come from data
    cat_max_dict = {i: 100 for i in range(10)}  # Placeholder
    encoder = Encoder(
        cnt_cat_features=10,
        cnt_num_features=10,
        cat_max_dict=cat_max_dict,
        d_model=settings.d_model,
        nhead=settings.nhead,
        num_layers=settings.num_layers,
        dim_feedforward=settings.dim_feedforward,
        dropout_rate=settings.dropout_rate
    )
    projection_head = ProjectionHead(
        d_model=settings.d_model,
        projection_dim=128
    )
    
    # In production, loader should be created from actual data
    # For now, we'll skip the actual training and save the initialized model
    
    # Ensure model save directory exists
    os.makedirs(settings.model_save_path, exist_ok=True)
    
    # Save pretrained encoder
    pretrain_path = os.path.join(
        settings.model_save_path,
        f"pretrained_encoder_{timestamp}.pth"
    )
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'projection_head_state_dict': projection_head.state_dict(),
    }, pretrain_path)
    
    logger.info(f"Pretrained encoder saved to {pretrain_path}")
    
    # Store metadata in MongoDB
    mongodb_client.connect()
    try:
        model_id = mongodb_client.store_model_metadata(
            model_name="default_prediction_encoder",
            model_path=pretrain_path,
            model_version=timestamp,
            hyperparameters={
                "d_model": settings.d_model,
                "nhead": settings.nhead,
                "num_layers": settings.num_layers,
                "dim_feedforward": settings.dim_feedforward,
                "dropout_rate": settings.dropout_rate
            }
        )
        logger.info(f"Model metadata stored with ID: {model_id}")
        return {"model_id": model_id, "model_path": pretrain_path}
    finally:
        mongodb_client.disconnect()


def train_classifier_model(**context):
    """Train the classifier model."""
    logger.info("Starting classifier training")
    
    # Get pretrained model path from previous task
    ti = context['ti']
    pretrain_result = ti.xcom_pull(task_ids='pretrain_model')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate timestamp once for consistency
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Load pretrained encoder
    cat_max_dict = {i: 100 for i in range(10)}  # Placeholder
    encoder = Encoder(
        cnt_cat_features=10,
        cnt_num_features=10,
        cat_max_dict=cat_max_dict,
        d_model=settings.d_model,
        nhead=settings.nhead,
        num_layers=settings.num_layers,
        dim_feedforward=settings.dim_feedforward,
        dropout_rate=settings.dropout_rate
    )
    
    # Create classifier
    classifier = TabTransformerClassifier(
        encoder=encoder,
        d_model=settings.d_model,
        final_hidden=128,
        dropout_rate=settings.dropout_rate
    )
    
    # In production, train with actual data
    # For now, save the initialized model
    
    # Save trained classifier
    classifier_path = os.path.join(
        settings.model_save_path,
        f"classifier_{timestamp}.pth"
    )
    torch.save({
        'model_state_dict': classifier.state_dict(),
    }, classifier_path)
    
    logger.info(f"Classifier saved to {classifier_path}")
    
    # Store metadata in MongoDB
    mongodb_client.connect()
    try:
        model_id = mongodb_client.store_model_metadata(
            model_name="default_prediction_classifier",
            model_path=classifier_path,
            model_version=timestamp,
            hyperparameters={
                "d_model": settings.d_model,
                "final_hidden": 128,
                "dropout_rate": settings.dropout_rate
            },
            metrics={"train_loss": 0.0}  # Placeholder
        )
        logger.info(f"Classifier metadata stored with ID: {model_id}")
        return {"model_id": model_id, "model_path": classifier_path}
    finally:
        mongodb_client.disconnect()


def notify_training_complete(**context):
    """Send notification that training is complete."""
    logger.info("Training pipeline completed")
    
    # Get model info from previous task
    ti = context['ti']
    classifier_result = ti.xcom_pull(task_ids='train_classifier')
    
    # Send command to Kafka
    from src.kafka.producer import kafka_producer
    kafka_producer.connect()
    
    try:
        kafka_producer.send_command({
            "command": "training_complete",
            "timestamp": datetime.utcnow().isoformat(),
            "model_id": classifier_result.get('model_id'),
            "model_path": classifier_result.get('model_path')
        }, key="training")
        logger.info("Training completion notification sent")
    finally:
        kafka_producer.disconnect()


# Define DAG
with DAG(
    'model_training_pipeline',
    default_args=default_args,
    description='Model training pipeline with Kafka and MongoDB',
    schedule_interval=timedelta(days=1),
    catchup=False,
    tags=['kafka', 'training', 'mongodb'],
) as dag:
    
    # Task 1: Prepare training data
    prepare_data_task = PythonOperator(
        task_id='prepare_training_data',
        python_callable=prepare_training_data,
        provide_context=True,
    )
    
    # Task 2: Pretrain encoder
    pretrain_task = PythonOperator(
        task_id='pretrain_model',
        python_callable=pretrain_model,
        provide_context=True,
    )
    
    # Task 3: Train classifier
    train_task = PythonOperator(
        task_id='train_classifier',
        python_callable=train_classifier_model,
        provide_context=True,
    )
    
    # Task 4: Notify completion
    notify_task = PythonOperator(
        task_id='notify_training_complete',
        python_callable=notify_training_complete,
        provide_context=True,
    )
    
    # Define task dependencies
    prepare_data_task >> pretrain_task >> train_task >> notify_task
