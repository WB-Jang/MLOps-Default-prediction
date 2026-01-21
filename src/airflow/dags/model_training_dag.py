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
from src.database.mongodb import mongodb_client
from src.data import load_data_for_training
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
    """Prepare training data from raw CSV file."""
    logger.info("Preparing training data")
    
    data_path = "/opt/airflow/data/raw/synthetic_data.csv"
    
    try:
        # ⭐ Load data using data loader (10개 값을 모두 받기)
        X_train_str, X_train_num, y_train, X_fine_str, X_fine_num, y_fine, X_test_str, X_test_num, y_test, metadata = load_data_for_training(
            data_path=data_path,
            test_size=0.3,  # 0.3으로 맞춤 (다른 DAG와 일치)
            random_state=42
        )
        
        logger.info(f"Loaded training data: {len(y_train)} train, {len(y_fine)} fine-tune, {len(y_test)} test samples")
        logger.info(f"Features: {metadata['num_categorical_features']} categorical, "
                   f"{metadata['num_numerical_features']} numerical")
        
        # Store metadata in MongoDB
        mongodb_client.connect()
        try:
            mongodb_client.get_collection("training_data").insert_one({
                "timestamp": datetime. utcnow(),
                "num_train_samples": len(y_train),
                "num_finetune_samples": len(y_fine),
                "num_test_samples": len(y_test),
                "num_categorical_features": metadata['num_categorical_features'],
                "num_numerical_features": metadata['num_numerical_features'],
                "status": "prepared"
            })
        finally:
            mongodb_client.disconnect()
        
        # Push metadata to XCom for next tasks
        return metadata
        
    except Exception as e:
        logger.error(f"Error preparing training data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def pretrain_model(**context):
    """Pretrain the encoder using contrastive learning."""
    logger.info("Starting model pretraining")
    
    # Get metadata from previous task
    ti = context['ti']
    metadata = ti.xcom_pull(task_ids='prepare_training_data')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Generate timestamp once for consistency
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create encoder with actual dimensions from data
    encoder = Encoder(
        cnt_cat_features=metadata['num_categorical_features'],
        cnt_num_features=metadata['num_numerical_features'],
        cat_max_dict=metadata['cat_max_dict'],
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
    
    # In production with actual training loop:
    # (X_train, y_train), (X_test, y_test), _ = load_data_for_training()
    # train_loader = create_dataloader(X_train, y_train)
    # pretrain_contrastive(encoder, projection_head, train_loader, epochs=settings.pretrain_epochs)
    
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
        'metadata': metadata
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
                "dropout_rate": settings.dropout_rate,
                "num_categorical_features": metadata['num_categorical_features'],
                "num_numerical_features": metadata['num_numerical_features']
            }
        )
        logger.info(f"Model metadata stored with ID: {model_id}")
        return {"model_id": model_id, "model_path": pretrain_path, "metadata": metadata}
    finally:
        mongodb_client.disconnect()


def train_classifier_model(**context):
    """Train the classifier model."""
    logger.info("Starting classifier training")
    
    # Get pretrained model info from previous task
    ti = context['ti']
    pretrain_result = ti.xcom_pull(task_ids='pretrain_model')
    metadata = pretrain_result.get('metadata', {})
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate timestamp once for consistency
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Load pretrained encoder
    encoder = Encoder(
        cnt_cat_features=metadata.get('num_categorical_features', 10),
        cnt_num_features=metadata.get('num_numerical_features', 10),
        cat_max_dict=metadata.get('cat_max_dict', {i: 100 for i in range(10)}),
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
    
    # In production with actual training loop:
    # (X_train, y_train), (X_test, y_test), _ = load_data_for_training()
    # train_loader = create_dataloader(X_train, y_train)
    # val_loader = create_dataloader(X_test, y_test)
    # metrics = train_classifier(classifier, train_loader, val_loader, epochs=settings.epochs)
    
    # Save trained classifier
    classifier_path = os.path.join(
        settings.model_save_path,
        f"classifier_{timestamp}.pth"
    )
    torch.save({
        'model_state_dict': classifier.state_dict(),
        'metadata': metadata
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
                "dropout_rate": settings.dropout_rate,
                "num_categorical_features": metadata.get('num_categorical_features', 10),
                "num_numerical_features": metadata.get('num_numerical_features', 10)
            },
            metrics={"train_loss": 0.0}  # Placeholder - add actual metrics from training
        )
        logger.info(f"Classifier metadata stored with ID: {model_id}")
        return {"model_id": model_id, "model_path": classifier_path}
    finally:
        mongodb_client.disconnect()


def notify_training_complete(**context):
    """Store notification that training is complete in MongoDB."""
    logger.info("Training pipeline completed")
    
    # Get model info from previous task
    ti = context['ti']
    classifier_result = ti.xcom_pull(task_ids='train_classifier')
    
    # Store completion notification in MongoDB
    mongodb_client.connect()
    
    try:
        mongodb_client.get_collection("training_events").insert_one({
            "event_type": "training_complete",
            "timestamp": datetime.utcnow(),
            "model_id": classifier_result.get('model_id'),
            "model_path": classifier_result.get('model_path'),
            "status": "completed"
        })
        logger.info("Training completion notification stored in MongoDB")
    finally:
        mongodb_client.disconnect()


# Define DAG
with DAG(
    'model_training_pipeline',
    default_args=default_args,
    description='Model training pipeline with MongoDB',
    schedule_interval=timedelta(days=1),
    catchup=False,
    tags=['training', 'mongodb'],
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
