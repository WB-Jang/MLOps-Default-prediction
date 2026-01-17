"""Airflow DAG for model deployment pipeline."""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.utils.dates import days_ago
import torch
import os
import shutil

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

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


def wait_for_model_ready(**context):
    """Check if a new approved model is available for deployment."""
    logger.info("Checking for approved model ready for deployment")
    
    mongodb_client.connect()
    
    try:
        # Get latest model with status 'approved' or 'retrained'
        collection = mongodb_client.get_collection("model_metadata")
        latest_model = collection.find_one(
            {"model_name": "default_prediction_classifier", "status": {"$in": ["approved", "retrained"]}},
            sort=[("created_at", -1)]
        )
        
        if not latest_model:
            raise ValueError("No approved model found for deployment")
        
        logger.info(f"Found approved model: {latest_model['model_version']}")
        
        return {
            "model_id": str(latest_model['_id']),
            "model_path": latest_model['model_path'],
            "model_version": latest_model['model_version']
        }
        
    finally:
        mongodb_client.disconnect()


def deploy_model(**context):
    """Deploy the approved model to production."""
    logger.info("Deploying model to production")
    
    ti = context['ti']
    model_info = ti.xcom_pull(task_ids='wait_for_model_ready')
    
    model_path = model_info['model_path']
    model_version = model_info['model_version']
    
    # Define deployment path
    deployment_dir = os.path.join(settings.model_save_path, "deployed")
    os.makedirs(deployment_dir, exist_ok=True)
    
    deployed_model_path = os.path.join(deployment_dir, "current_model.pth")
    
    try:
        # Copy model to deployment location
        shutil.copy2(model_path, deployed_model_path)
        logger.info(f"Model copied to deployment location: {deployed_model_path}")
        
        # Create version symlink or metadata file
        version_file = os.path.join(deployment_dir, "model_version.txt")
        with open(version_file, 'w') as f:
            f.write(f"Version: {model_version}\n")
            f.write(f"Deployed at: {datetime.utcnow().isoformat()}\n")
            f.write(f"Model ID: {model_info['model_id']}\n")
        
        logger.info(f"Model version {model_version} deployed successfully")
        
        # Update deployment status in MongoDB
        mongodb_client.connect()
        try:
            # Store deployment event
            deployment_collection = mongodb_client.get_collection("deployment_events")
            deployment_collection.insert_one({
                "event_type": "model_deployed",
                "timestamp": datetime.utcnow(),
                "model_id": model_info['model_id'],
                "model_version": model_version,
                "deployment_path": deployed_model_path,
                "status": "deployed"
            })
            
            # Update model metadata status
            mongodb_client.update_model_status(
                model_id=model_info['model_id'],
                status="deployed",
                notes=f"Deployed to {deployed_model_path} at {datetime.utcnow().isoformat()}"
            )
            
            logger.info("Deployment event stored in MongoDB")
            
        finally:
            mongodb_client.disconnect()
        
        return {
            "model_id": model_info['model_id'],
            "model_version": model_version,
            "deployment_path": deployed_model_path,
            "deployment_timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error during model deployment: {e}")
        
        # Store deployment failure
        mongodb_client.connect()
        try:
            deployment_collection = mongodb_client.get_collection("deployment_events")
            deployment_collection.insert_one({
                "event_type": "deployment_failed",
                "timestamp": datetime.utcnow(),
                "model_id": model_info['model_id'],
                "model_version": model_version,
                "error": str(e),
                "status": "failed"
            })
        finally:
            mongodb_client.disconnect()
        
        raise


def notify_deployment_complete(**context):
    """Send notification that deployment is complete."""
    logger.info("Deployment pipeline completed")
    
    ti = context['ti']
    deployment_info = ti.xcom_pull(task_ids='deploy_model')
    
    logger.info(f"Model {deployment_info['model_version']} successfully deployed")
    logger.info(f"Deployment path: {deployment_info['deployment_path']}")
    logger.info(f"Timestamp: {deployment_info['deployment_timestamp']}")
    
    # Store notification
    mongodb_client.connect()
    try:
        notification_collection = mongodb_client.get_collection("notifications")
        notification_collection.insert_one({
            "type": "deployment_complete",
            "timestamp": datetime.utcnow(),
            "model_id": deployment_info['model_id'],
            "model_version": deployment_info['model_version'],
            "message": f"Model {deployment_info['model_version']} deployed successfully"
        })
        
        # TODO: Send actual notifications (email, Slack, etc.)
        # Example: send_slack_notification(deployment_info)
        
    finally:
        mongodb_client.disconnect()


# Define DAG
with DAG(
    'model_deployment_pipeline',
    default_args=default_args,
    description='Model deployment pipeline',
    schedule_interval=timedelta(days=1),
    catchup=False,
    tags=['deployment', 'mongodb'],
) as dag:
    
    # Task 1: Wait for evaluation to complete (external task sensor)
    wait_for_evaluation = ExternalTaskSensor(
        task_id='wait_for_evaluation',
        external_dag_id='model_evaluation_pipeline',
        external_task_id='send_evaluation_results',
        allowed_states=['success'],
        failed_states=['failed', 'skipped'],
        mode='reschedule',
        timeout=3600,  # 1 hour timeout
        poke_interval=300,  # Check every 5 minutes
    )
    
    # Task 2: Check for approved model
    check_model_task = PythonOperator(
        task_id='wait_for_model_ready',
        python_callable=wait_for_model_ready,
        provide_context=True,
    )
    
    # Task 3: Deploy model
    deploy_task = PythonOperator(
        task_id='deploy_model',
        python_callable=deploy_model,
        provide_context=True,
    )
    
    # Task 4: Notify deployment complete
    notify_task = PythonOperator(
        task_id='notify_deployment_complete',
        python_callable=notify_deployment_complete,
        provide_context=True,
    )
    
    # Define task dependencies
    wait_for_evaluation >> check_model_task >> deploy_task >> notify_task
