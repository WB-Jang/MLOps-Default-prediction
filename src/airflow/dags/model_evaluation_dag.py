"""Airflow DAG for model evaluation pipeline with retry mechanism."""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.utils.dates import days_ago
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.database.mongodb import mongodb_client
from src.models import Encoder, TabTransformerClassifier
from src.models.evaluation import evaluate_model as eval_model_func, finetune_model as finetune_func
from src.data.data_gen_loader_processor import load_data_for_training
from config.settings import settings
from loguru import logger


# F1 score threshold for retraining
F1_THRESHOLD = 0.75
MAX_RETRAIN_ATTEMPTS = 3


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
    """Evaluate the model on test data using evaluation functions from corp_default_modeling_f.py."""
    logger.info("Evaluating model using standardized evaluation function")
    
    # Get model info from previous task
    ti = context['ti']
    model_info = ti.xcom_pull(task_ids='load_latest_model')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Load data for evaluation
        data_path = "./data/raw/synthetic_data.csv"
        X_train_str, X_train_num, y_train, X_fine_str, X_fine_num, y_fine, X_test_str, X_test_num, y_test, metadata = load_data_for_training(
            data_path=data_path,
            test_size=0.3,
            random_state=42
        )
        
        # Create DataLoader for test data
        class DataLoading(Dataset):
            def __init__(self, X_str, X_num, y):
                self.X_str = X_str
                self.X_num = X_num
                self.y = y
            
            def __len__(self):
                return len(self.y)
            
            def __getitem__(self, idx):
                return {
                    "str": torch.tensor(self.X_str[idx], dtype=torch.long),
                    "num": torch.tensor(self.X_num[idx], dtype=torch.float),
                    "label": torch.tensor(self.y[idx], dtype=torch.long)
                }
        
        test_dataset = DataLoading(X_test_str, X_test_num, y_test)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        # Load model
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
        
        classifier = TabTransformerClassifier(
            encoder=encoder,
            d_model=settings.d_model,
            final_hidden=128,
            dropout_rate=settings.dropout_rate
        )
        
        checkpoint = torch.load(model_info['model_path'], map_location=device)
        classifier.load_state_dict(checkpoint['model_state_dict'])
        
        # Use evaluation function from corp_default_modeling_f.py
        metrics = eval_model_func(classifier, test_loader, device=device)
        
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
                "metrics": metrics,
                "metadata": metadata,
                "data_path": data_path
            }
            
        finally:
            mongodb_client.disconnect()
            
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise


def check_model_performance(**context):
    """Check if model performance meets threshold and decide on retraining."""
    logger.info("Checking model performance")
    
    # Get evaluation results
    ti = context['ti']
    eval_result = ti.xcom_pull(task_ids='evaluate_model')
    
    metrics = eval_result['metrics']
    
    # Get retry attempt count from XCom (default to 0)
    retry_count = ti.xcom_pull(key='retry_count', default=0)
    
    # Check F1 score against threshold
    needs_retraining = metrics['f1_score'] < F1_THRESHOLD
    
    if needs_retraining:
        logger.warning(f"Model F1 score {metrics['f1_score']:.4f} below threshold {F1_THRESHOLD}")
        logger.info(f"Retraining attempt {retry_count + 1}/{MAX_RETRAIN_ATTEMPTS}")
        
        if retry_count < MAX_RETRAIN_ATTEMPTS:
            status = "needs_retraining"
            next_task = 'prepare_retraining_data'
            # Increment retry count
            ti.xcom_push(key='retry_count', value=retry_count + 1)
        else:
            logger.error(f"Maximum retraining attempts ({MAX_RETRAIN_ATTEMPTS}) reached. Sending alert.")
            status = "failed_max_retries"
            next_task = 'send_alert'
    else:
        logger.info(f"Model performance meets requirements (F1: {metrics['f1_score']:.4f})")
        status = "approved"
        next_task = 'send_evaluation_results'
        # Reset retry count on success
        ti.xcom_push(key='retry_count', value=0)
    
    # Update model status in MongoDB
    mongodb_client.connect()
    
    try:
        mongodb_client.update_model_status(
            model_id=eval_result['model_id'],
            status=status,
            notes=f"F1 Score: {metrics['f1_score']:.4f}, Threshold: {F1_THRESHOLD}, Retry: {retry_count}"
        )
    finally:
        mongodb_client.disconnect()
    
    # Store decision for later tasks
    ti.xcom_push(key='performance_check', value={
        "status": status,
        "metrics": metrics,
        "needs_retraining": needs_retraining,
        "f1_threshold": F1_THRESHOLD,
        "retry_count": retry_count
    })
    
    return next_task


def prepare_retraining_data(**context):
    """Prepare data for retraining by loading and splitting evaluation data."""
    logger.info("Preparing data for retraining")
    
    try:
        # Load the full dataset
        data_path = "./data/raw/synthetic_data.csv"
        (X_train, y_train), (X_test, y_test), metadata = load_data_for_training(
            data_path=data_path,
            test_size=0.2,
            random_state=42
        )
        
        # Use test data for retraining (split into retrain/revalidation)
        # This simulates using fresh evaluation data for fine-tuning
        X_retrain_str = X_test["str"][:int(len(X_test["str"]) * 0.7)]
        X_retrain_num = X_test["num"][:int(len(X_test["num"]) * 0.7)]
        y_retrain = y_test[:int(len(y_test) * 0.7)]
        
        X_reval_str = X_test["str"][int(len(X_test["str"]) * 0.7):]
        X_reval_num = X_test["num"][int(len(X_test["num"]) * 0.7):]
        y_reval = y_test[int(len(y_test) * 0.7):]
        
        logger.info(f"Retraining data: {len(y_retrain)} samples, Re-evaluation: {len(y_reval)} samples")
        
        # Store data info in XCom (we'll pass metadata and data info)
        return {
            "retrain_samples": len(y_retrain),
            "reval_samples": len(y_reval),
            "metadata": metadata,
            "data_path": data_path
        }
        
    except Exception as e:
        logger.error(f"Error preparing retraining data: {e}")
        raise


def finetune_model(**context):
    """Fine-tune the classifier using code from corp_default_modeling_f.py."""
    logger.info("Starting model fine-tuning using standardized fine-tuning function")
    
    ti = context['ti']
    model_info = ti.xcom_pull(task_ids='load_latest_model')
    eval_result = ti.xcom_pull(task_ids='evaluate_model')
    retrain_info = ti.xcom_pull(task_ids='prepare_retraining_data')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        # Load the data for fine-tuning
        data_path = eval_result['data_path']
        metadata = eval_result['metadata']
        X_train_str, X_train_num, y_train, X_fine_str, X_fine_num, y_fine, X_test_str, X_test_num, y_test, _ = load_data_for_training(
            data_path=data_path,
            test_size=0.3,
            random_state=42
        )
        
        # Create Dataset class
        class DataLoading(Dataset):
            def __init__(self, X_str, X_num, y):
                self.X_str = X_str
                self.X_num = X_num
                self.y = y
            
            def __len__(self):
                return len(self.y)
            
            def __getitem__(self, idx):
                return {
                    "str": torch.tensor(self.X_str[idx], dtype=torch.long),
                    "num": torch.tensor(self.X_num[idx], dtype=torch.float),
                    "label": torch.tensor(self.y[idx], dtype=torch.long)
                }
        
        # Use fine-tuning data (evaluation data)
        fine_dataset = DataLoading(X_fine_str, X_fine_num, y_fine)
        test_dataset = DataLoading(X_test_str, X_test_num, y_test)
        
        fine_loader = DataLoader(fine_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        # Load the existing model
        model_path = model_info['model_path']
        logger.info(f"Loading model from {model_path}")
        
        # Create model architecture
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
        
        classifier = TabTransformerClassifier(
            encoder=encoder,
            d_model=settings.d_model,
            final_hidden=128,
            dropout_rate=settings.dropout_rate
        )
        
        # Load existing weights
        checkpoint = torch.load(model_path, map_location=device)
        classifier.load_state_dict(checkpoint['model_state_dict'])
        classifier.to(device)
        
        # Compute class weights for balanced training
        classes = np.unique(y_fine)
        class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_fine)
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        
        # Fine-tuning with class weights (as in corp_default_modeling_f.py)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Use finetune function from corp_default_modeling_f.py
        training_metrics, test_metrics = finetune_func(
            model=classifier,
            train_loader=fine_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=3
        )
        
        # Save fine-tuned model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        finetuned_path = os.path.join(
            settings.model_save_path,
            f"classifier_finetuned_{timestamp}.pth"
        )
        
        os.makedirs(settings.model_save_path, exist_ok=True)
        torch.save({
            'model_state_dict': classifier.state_dict(),
            'metadata': metadata,
            'finetuned_from': model_info['model_id'],
            'finetune_timestamp': timestamp
        }, finetuned_path)
        
        logger.info(f"Fine-tuned model saved to {finetuned_path}")
        
        # Store metadata in MongoDB
        mongodb_client.connect()
        try:
            finetuned_model_id = mongodb_client.store_model_metadata(
                model_name="default_prediction_classifier",
                model_path=finetuned_path,
                model_version=timestamp,
                hyperparameters={
                    "d_model": settings.d_model,
                    "final_hidden": 128,
                    "dropout_rate": settings.dropout_rate,
                    "num_categorical_features": metadata['num_categorical_features'],
                    "num_numerical_features": metadata['num_numerical_features'],
                    "finetuned_from": model_info['model_id'],
                    "finetune_epochs": 3,
                    "finetune_lr": 1e-4
                },
                metrics=test_metrics
            )
            logger.info(f"Fine-tuned model metadata stored with ID: {finetuned_model_id}")
            
            return {
                "model_id": finetuned_model_id,
                "model_path": finetuned_path,
                "finetuned_from": model_info['model_id'],
                "metadata": metadata,
                "data_path": data_path,
                "test_metrics": test_metrics
            }
        finally:
            mongodb_client.disconnect()
            
    except Exception as e:
        logger.error(f"Error during fine-tuning: {e}")
        raise


def evaluate_finetuned_model(**context):
    """Evaluate the fine-tuned model and check if it needs another retry."""
    logger.info("Evaluating fine-tuned model")
    
    ti = context['ti']
    finetuned_info = ti.xcom_pull(task_ids='finetune_model')
    
    # Get test metrics from fine-tuning
    test_metrics = finetuned_info['test_metrics']
    
    logger.info(f"Fine-tuned model metrics: {test_metrics}")
    
    # Get retry count
    retry_count = ti.xcom_pull(key='retry_count', default=1)
    
    # Check if performance is still below threshold
    if test_metrics['f1_score'] < F1_THRESHOLD:
        logger.warning(
            f"Fine-tuned model F1 score {test_metrics['f1_score']:.4f} "
            f"still below threshold {F1_THRESHOLD}. Retry {retry_count}/{MAX_RETRAIN_ATTEMPTS}"
        )
        
        if retry_count < MAX_RETRAIN_ATTEMPTS:
            # Need to retry - loop back to evaluation
            next_task = 'check_model_performance'
            logger.info(f"Will retry retraining (attempt {retry_count + 1}/{MAX_RETRAIN_ATTEMPTS})")
        else:
            # Max retries reached - send alert
            next_task = 'send_alert'
            logger.error(f"Maximum retraining attempts reached. F1 score still {test_metrics['f1_score']:.4f}")
    else:
        # Performance is good - continue to results
        next_task = 'send_evaluation_results'
        logger.info(f"Fine-tuned model meets threshold. F1 score: {test_metrics['f1_score']:.4f}")
    
    # Store evaluation in MongoDB
    mongodb_client.connect()
    try:
        metrics_id = mongodb_client.store_performance_metrics(
            model_id=finetuned_info['model_id'],
            metrics=test_metrics,
            dataset_name="post_finetune_test"
        )
        logger.info(f"Fine-tuned model metrics stored with ID: {metrics_id}")
        
        return {
            "model_id": finetuned_info['model_id'],
            "metrics": test_metrics,
            "finetuned_from": finetuned_info['finetuned_from'],
            "next_task": next_task
        }
    finally:
        mongodb_client.disconnect()


def send_alert(**context):
    """Send alert when model fails to meet threshold after maximum retries."""
    logger.error("ALERT: Model failed to meet performance threshold after maximum retries")
    
    ti = context['ti']
    model_info = ti.xcom_pull(task_ids='load_latest_model')
    performance_check = ti.xcom_pull(key='performance_check', task_ids='check_model_performance')
    retry_count = ti.xcom_pull(key='retry_count', default=MAX_RETRAIN_ATTEMPTS)
    
    # Get latest finetuned metrics if available
    finetuned_result = ti.xcom_pull(task_ids='evaluate_finetuned_model')
    
    if finetuned_result:
        final_f1 = finetuned_result['metrics']['f1_score']
        final_model_id = finetuned_result['model_id']
    else:
        final_f1 = performance_check['metrics']['f1_score']
        final_model_id = model_info['model_id']
    
    alert_message = {
        "alert_type": "MODEL_PERFORMANCE_FAILURE",
        "severity": "CRITICAL",
        "timestamp": datetime.utcnow(),
        "model_id": final_model_id,
        "original_model_id": model_info['model_id'],
        "f1_score": final_f1,
        "f1_threshold": F1_THRESHOLD,
        "retry_attempts": retry_count,
        "max_retries": MAX_RETRAIN_ATTEMPTS,
        "message": (
            f"Model failed to meet F1 threshold of {F1_THRESHOLD} after {retry_count} retraining attempts. "
            f"Final F1 score: {final_f1:.4f}. Manual intervention required."
        )
    }
    
    logger.error(f"Alert details: {alert_message}")
    
    # Store alert in MongoDB
    mongodb_client.connect()
    try:
        alert_collection = mongodb_client.get_collection("alerts")
        alert_id = alert_collection.insert_one(alert_message).inserted_id
        logger.error(f"Alert stored in MongoDB with ID: {alert_id}")
        
        # Also store in evaluation events for tracking
        event_document = {
            "event_type": "evaluation_failed",
            "timestamp": datetime.utcnow(),
            "model_id": final_model_id,
            "status": "failed_max_retries",
            "metrics": {"f1_score": final_f1},
            "retry_attempts": retry_count,
            "alert_id": str(alert_id)
        }
        mongodb_client.get_collection("evaluation_events").insert_one(event_document)
        
        # TODO: Integrate with actual alerting system (email, Slack, PagerDuty, etc.)
        # Example: send_email_alert(alert_message)
        # Example: send_slack_alert(alert_message)
        
    finally:
        mongodb_client.disconnect()
    
    logger.error("Alert sent successfully. Manual intervention required.")


def send_evaluation_results(**context):
    """Store evaluation results in MongoDB."""
    logger.info("Storing evaluation results in MongoDB")
    
    # Get evaluation status
    ti = context['ti']
    performance_check = ti.xcom_pull(key='performance_check', task_ids='check_model_performance')
    model_info = ti.xcom_pull(task_ids='load_latest_model')
    
    # Check if retraining was performed
    finetuned_result = ti.xcom_pull(task_ids='evaluate_finetuned_model')
    
    # Determine final status and metrics
    if finetuned_result:
        # Retraining was performed
        final_status = "retrained"
        final_metrics = finetuned_result['metrics']
        final_model_id = finetuned_result['model_id']
        original_model_id = model_info['model_id']
        retraining_info = {
            "was_retrained": True,
            "original_model_id": original_model_id,
            "original_f1_score": performance_check['metrics']['f1_score'],
            "finetuned_model_id": final_model_id,
            "finetuned_f1_score": final_metrics['f1_score'],
            "improvement": final_metrics['f1_score'] - performance_check['metrics']['f1_score']
        }
        logger.info(f"Model was retrained. F1 improvement: {retraining_info['improvement']:.4f}")
    else:
        # No retraining needed
        final_status = performance_check['status']
        final_metrics = performance_check['metrics']
        final_model_id = model_info['model_id']
        retraining_info = {
            "was_retrained": False,
            "f1_score": final_metrics['f1_score'],
            "f1_threshold": performance_check['f1_threshold']
        }
        logger.info("Model performance was satisfactory. No retraining needed.")
    
    # Connect to MongoDB
    mongodb_client.connect()
    
    try:
        # Store evaluation event
        event_document = {
            "event_type": "evaluation_complete",
            "timestamp": datetime.utcnow(),
            "model_id": final_model_id,
            "model_version": model_info['model_version'],
            "status": final_status,
            "metrics": final_metrics,
            "retraining_info": retraining_info
        }
        
        mongodb_client.get_collection("evaluation_events").insert_one(event_document)
        
        logger.info(f"Evaluation results stored in MongoDB. Status: {final_status}")
        logger.info(f"Final metrics: {final_metrics}")
        
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
    
    # Task 3: Check performance (branching)
    check_task = BranchPythonOperator(
        task_id='check_model_performance',
        python_callable=check_model_performance,
        provide_context=True,
    )
    
    # Task 4a: Prepare retraining data (if performance is low)
    prepare_retrain_task = PythonOperator(
        task_id='prepare_retraining_data',
        python_callable=prepare_retraining_data,
        provide_context=True,
    )
    
    # Task 4b: Fine-tune model
    finetune_task = PythonOperator(
        task_id='finetune_model',
        python_callable=finetune_model,
        provide_context=True,
    )
    
    # Task 4c: Evaluate fine-tuned model (returns branch decision)
    evaluate_finetuned_task = BranchPythonOperator(
        task_id='evaluate_finetuned_model',
        python_callable=evaluate_finetuned_model,
        provide_context=True,
    )
    
    # Task 5a: Send alert (if max retries exceeded)
    send_alert_task = PythonOperator(
        task_id='send_alert',
        python_callable=send_alert,
        provide_context=True,
    )
    
    # Task 5b: Send results (both branches converge here)
    send_results_task = PythonOperator(
        task_id='send_evaluation_results',
        python_callable=send_evaluation_results,
        provide_context=True,
        trigger_rule='none_failed_min_one_success',  # Run if at least one upstream task succeeds
    )
    
    # Define task dependencies
    # Main flow
    load_model_task >> evaluate_task >> check_task
    
    # Branch 1: Retraining path (if F1 < threshold)
    check_task >> prepare_retrain_task >> finetune_task >> evaluate_finetuned_task
    
    # Branch 1a: Retry loop (if still below threshold and retries remain)
    # Note: evaluate_finetuned_task returns 'check_model_performance' to loop back
    evaluate_finetuned_task >> check_task  # Loop back for retry
    
    # Branch 1b: Success after retraining
    evaluate_finetuned_task >> send_results_task
    
    # Branch 2: Direct to results (if F1 >= threshold)
    check_task >> send_results_task
    
    # Branch 3: Alert path (if max retries exceeded)
    check_task >> send_alert_task
    evaluate_finetuned_task >> send_alert_task
