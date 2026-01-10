"""Airflow DAG for model evaluation pipeline."""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import torch

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.database.mongodb import mongodb_client
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


def load_latest_model(**context):
    """Load the latest model from MongoDB."""
    logger.info("Loading latest model")
    
    mongodb_client.connect()
    
    try:
        # Get latest classifier model
        model_metadata = mongodb_client.get_model_metadata(
            model_name="default_prediction_classifier"
        )
        
        if not model_metadata:
            raise ValueError("No model found in database")
        
        logger.info(f"Found model: {model_metadata['model_path']}")
        
        return {
            "model_id": str(model_metadata['_id']),
            "model_path": model_metadata['model_path'],
            "model_version": model_metadata['model_version']
        }
        
    finally:
        mongodb_client.disconnect()


def evaluate_model(**context):
    """Evaluate the model on test data."""
    logger.info("Evaluating model")
    
    # Get model info from previous task
    ti = context['ti']
    model_info = ti.xcom_pull(task_ids='load_latest_model')
    
    # In production, this would load actual test data and evaluate
    # For demonstration, creating sample metrics
    metrics = {
        "accuracy": 0.85,
        "precision": 0.82,
        "recall": 0.88,
        "f1_score": 0.85,
        "roc_auc": 0.90
    }
    
    logger.info(f"Model evaluation metrics: {metrics}")
    
    # Store metrics in MongoDB
    mongodb_client.connect()
    
    try:
        metrics_id = mongodb_client.store_performance_metrics(
            model_id=model_info['model_id'],
            metrics=metrics,
            dataset_name="test"
        )
        logger.info(f"Metrics stored with ID: {metrics_id}")
        
        return {
            "model_id": model_info['model_id'],
            "metrics": metrics
        }
        
    finally:
        mongodb_client.disconnect()


def check_model_performance(**context):
    """Check if model performance meets threshold."""
    logger.info("Checking model performance")
    
    # Get evaluation results
    ti = context['ti']
    eval_result = ti.xcom_pull(task_ids='evaluate_model')
    
    metrics = eval_result['metrics']
    
    # Define performance thresholds
    accuracy_threshold = 0.80
    f1_threshold = 0.75
    
    if (metrics['accuracy'] >= accuracy_threshold and 
        metrics['f1_score'] >= f1_threshold):
        logger.info("Model performance meets requirements")
        status = "approved"
    else:
        logger.warning("Model performance below threshold")
        status = "rejected"
    
    # Update model status in MongoDB
    mongodb_client.connect()
    
    try:
        mongodb_client.update_model_status(
            model_id=eval_result['model_id'],
            status=status,
            notes=f"Accuracy: {metrics['accuracy']:.2f}, F1: {metrics['f1_score']:.2f}"
        )
    finally:
        mongodb_client.disconnect()
    
    return {"status": status, "metrics": metrics}


def send_evaluation_results(**context):
    """Store evaluation results in MongoDB."""
    logger.info("Storing evaluation results in MongoDB")
    
    # Get evaluation status
    ti = context['ti']
    check_result = ti.xcom_pull(task_ids='check_model_performance')
    model_info = ti.xcom_pull(task_ids='load_latest_model')
    
    # Connect to MongoDB
    mongodb_client.connect()
    
    try:
        # Store evaluation event
        mongodb_client.get_collection("evaluation_events").insert_one({
            "event_type": "evaluation_complete",
            "timestamp": datetime.utcnow(),
            "model_id": model_info['model_id'],
            "model_version": model_info['model_version'],
            "status": check_result['status'],
            "metrics": check_result['metrics']
        })
        
        logger.info("Evaluation results stored in MongoDB")
        
    finally:
        mongodb_client.disconnect()


# Define DAG
with DAG(
    'model_evaluation_pipeline',
    default_args=default_args,
    description='Model evaluation pipeline with MongoDB',
    schedule_interval=timedelta(days=1),
    catchup=False,
    tags=['evaluation', 'mongodb'],
) as dag:
    
    # Task 1: Load latest model
    load_model_task = PythonOperator(
        task_id='load_latest_model',
        python_callable=load_latest_model,
        provide_context=True,
    )
    
    # Task 2: Evaluate model
    evaluate_task = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model,
        provide_context=True,
    )
    
    # Task 3: Check performance
    check_task = PythonOperator(
        task_id='check_model_performance',
        python_callable=check_model_performance,
        provide_context=True,
    )
    
    # Task 4: Send results
    send_results_task = PythonOperator(
        task_id='send_evaluation_results',
        python_callable=send_evaluation_results,
        provide_context=True,
    )
    
    # Define task dependencies
    load_model_task >> evaluate_task >> check_task >> send_results_task
