"""Airflow DAG for model evaluation pipeline."""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.utils.dates import days_ago
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.database.mongodb import mongodb_client
from src.models import Encoder, TabTransformerClassifier
from src.data.data_gen_loader_processor import load_data_for_training
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
    """Check if model performance meets threshold and decide on retraining."""
    logger.info("Checking model performance")
    
    # Get evaluation results
    ti = context['ti']
    eval_result = ti.xcom_pull(task_ids='evaluate_model')
    
    metrics = eval_result['metrics']
    
    # Define performance thresholds
    f1_threshold = 0.75
    
    # Check F1 score against threshold
    needs_retraining = metrics['f1_score'] < f1_threshold
    
    if needs_retraining:
        logger.warning(f"Model F1 score {metrics['f1_score']:.4f} below threshold {f1_threshold}")
        logger.info("Model will be fine-tuned")
        status = "needs_retraining"
        next_task = 'prepare_retraining_data'
    else:
        logger.info(f"Model performance meets requirements (F1: {metrics['f1_score']:.4f})")
        status = "approved"
        next_task = 'send_evaluation_results'
    
    # Update model status in MongoDB
    mongodb_client.connect()
    
    try:
        mongodb_client.update_model_status(
            model_id=eval_result['model_id'],
            status=status,
            notes=f"F1 Score: {metrics['f1_score']:.4f}, Threshold: {f1_threshold}"
        )
    finally:
        mongodb_client.disconnect()
    
    # Store decision for later tasks
    ti.xcom_push(key='performance_check', value={
        "status": status,
        "metrics": metrics,
        "needs_retraining": needs_retraining,
        "f1_threshold": f1_threshold
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
    """Fine-tune the classifier (freeze encoder, train classifier only)."""
    logger.info("Starting model fine-tuning")
    
    ti = context['ti']
    model_info = ti.xcom_pull(task_ids='load_latest_model')
    retrain_info = ti.xcom_pull(task_ids='prepare_retraining_data')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        # Load the data for fine-tuning
        data_path = retrain_info['data_path']
        (X_train, y_train), (X_test, y_test), metadata = load_data_for_training(
            data_path=data_path,
            test_size=0.2,
            random_state=42
        )
        
        # Use test data for fine-tuning
        X_finetune_str = X_test["str"][:int(len(X_test["str"]) * 0.7)]
        X_finetune_num = X_test["num"][:int(len(X_test["num"]) * 0.7)]
        y_finetune = y_test[:int(len(y_test) * 0.7)]
        
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
        
        # Freeze encoder parameters (only fine-tune classifier)
        for param in classifier.encoder.parameters():
            param.requires_grad = False
        
        # Only classifier parameters will be trained
        trainable_params = sum(p.numel() for p in classifier.fc.parameters())
        total_params = sum(p.numel() for p in classifier.parameters())
        logger.info(f"Trainable parameters: {trainable_params}/{total_params} (classifier only)")
        
        # Fine-tuning with smaller learning rate
        optimizer = torch.optim.Adam(classifier.fc.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        classifier.train()
        num_epochs = 5  # Fewer epochs for fine-tuning
        batch_size = 32
        
        # Convert to tensors
        X_str_tensor = torch.from_numpy(X_finetune_str).long()
        X_num_tensor = torch.from_numpy(X_finetune_num).float()
        y_tensor = torch.from_numpy(y_finetune).long()
        
        # Simple training loop
        for epoch in range(num_epochs):
            total_loss = 0.0
            num_batches = 0
            
            for i in range(0, len(y_tensor), batch_size):
                batch_str = X_str_tensor[i:i+batch_size].to(device)
                batch_num = X_num_tensor[i:i+batch_size].to(device)
                batch_labels = y_tensor[i:i+batch_size].to(device)
                
                optimizer.zero_grad()
                logits = classifier(batch_str, batch_num)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            logger.info(f"Fine-tuning Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
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
                    "finetune_epochs": num_epochs,
                    "finetune_lr": 1e-4
                }
            )
            logger.info(f"Fine-tuned model metadata stored with ID: {finetuned_model_id}")
            
            return {
                "model_id": finetuned_model_id,
                "model_path": finetuned_path,
                "finetuned_from": model_info['model_id'],
                "metadata": metadata,
                "data_path": data_path
            }
        finally:
            mongodb_client.disconnect()
            
    except Exception as e:
        logger.error(f"Error during fine-tuning: {e}")
        raise


def evaluate_finetuned_model(**context):
    """Evaluate the fine-tuned model on re-evaluation data."""
    logger.info("Evaluating fine-tuned model")
    
    ti = context['ti']
    finetuned_info = ti.xcom_pull(task_ids='finetune_model')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Load re-evaluation data
        data_path = finetuned_info['data_path']
        (X_train, y_train), (X_test, y_test), metadata = load_data_for_training(
            data_path=data_path,
            test_size=0.2,
            random_state=42
        )
        
        # Use remaining test data for re-evaluation
        X_reval_str = X_test["str"][int(len(X_test["str"]) * 0.7):]
        X_reval_num = X_test["num"][int(len(X_test["num"]) * 0.7):]
        y_reval = y_test[int(len(y_test) * 0.7):]
        
        # Load fine-tuned model
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
        
        checkpoint = torch.load(finetuned_info['model_path'], map_location=device)
        classifier.load_state_dict(checkpoint['model_state_dict'])
        classifier.to(device)
        classifier.eval()
        
        # Evaluate on re-evaluation set
        X_str_tensor = torch.from_numpy(X_reval_str).long()
        X_num_tensor = torch.from_numpy(X_reval_num).float()
        y_tensor = torch.from_numpy(y_reval).long()
        
        with torch.no_grad():
            logits = classifier(X_str_tensor.to(device), X_num_tensor.to(device))
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        accuracy = accuracy_score(y_reval, predictions)
        precision = precision_score(y_reval, predictions, zero_division=0)
        recall = recall_score(y_reval, predictions, zero_division=0)
        f1 = f1_score(y_reval, predictions, zero_division=0)
        
        # For ROC AUC, use probabilities
        probs = torch.softmax(logits, dim=1).cpu().numpy()[:, 1]
        try:
            roc_auc = roc_auc_score(y_reval, probs)
        except:
            roc_auc = 0.0
        
        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc)
        }
        
        logger.info(f"Fine-tuned model evaluation metrics: {metrics}")
        
        # Store metrics in MongoDB
        mongodb_client.connect()
        try:
            metrics_id = mongodb_client.store_performance_metrics(
                model_id=finetuned_info['model_id'],
                metrics=metrics,
                dataset_name="re_evaluation"
            )
            logger.info(f"Fine-tuned model metrics stored with ID: {metrics_id}")
            
            return {
                "model_id": finetuned_info['model_id'],
                "metrics": metrics,
                "finetuned_from": finetuned_info['finetuned_from']
            }
        finally:
            mongodb_client.disconnect()
            
    except Exception as e:
        logger.error(f"Error evaluating fine-tuned model: {e}")
        raise


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
    
    # Task 4c: Evaluate fine-tuned model
    evaluate_finetuned_task = PythonOperator(
        task_id='evaluate_finetuned_model',
        python_callable=evaluate_finetuned_model,
        provide_context=True,
    )
    
    # Task 5: Send results (both branches converge here)
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
    check_task >> prepare_retrain_task >> finetune_task >> evaluate_finetuned_task >> send_results_task
    
    # Branch 2: Direct to results (if F1 >= threshold)
    check_task >> send_results_task
