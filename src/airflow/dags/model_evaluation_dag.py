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
    """Evaluate the model on test data using evaluation functions."""
    import torch
    import numpy as np
    from torch.utils.data import DataLoader, Dataset
    import torch.nn as nn

    logger.info("Evaluating model using standardized evaluation function")
    
    ti = context['ti']
    model_info = ti.xcom_pull(task_ids='load_latest_model')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_info is None: 
        raise ValueError("No model info from load_latest_model.")
    
    # 1. 체크포인트 로드 및 구조 분석
    logger.info(f"Loading checkpoint from {model_info['model_path']}")
    checkpoint = torch.load(model_info['model_path'], map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # [FIX 1] Key 이름 매핑
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("numeric_embeddings", "num_embeddings")
        new_state_dict[new_key] = v
    state_dict = new_state_dict

    # [FIX 2] Vocab Size 및 Feature 개수 역추적
    ckpt_cat_max_dict = {}
    max_num_idx = -1
    
    for k, v in state_dict.items():
        if "encoder.embeddings." in k and ".weight" in k:
            try:
                idx = int(k.split('.')[2])
                ckpt_cat_max_dict[idx] = v.shape[0]
            except: pass
        if "encoder.num_embeddings." in k:
            try:
                idx = int(k.split('.')[2])
                if idx > max_num_idx: max_num_idx = idx
            except: pass

    # 모델이 기대하는 Feature 개수 계산
    expected_cat_feats = max(ckpt_cat_max_dict.keys()) + 1 if ckpt_cat_max_dict else 0
    expected_num_feats = max_num_idx + 1
    
    logger.info(f"✅ Model expects: {expected_cat_feats} categorical, {expected_num_feats} numerical features")

    # 2. 데이터 로드
    try:
        data_path = "./data/raw/synthetic_data.csv"
        X_train_str, X_train_num, y_train, X_fine_str, X_fine_num, y_fine, X_test_str, X_test_num, y_test, metadata = load_data_for_training(
            data_path=data_path,
            test_size=0.3,
            random_state=42
        )
        
        # ⭐ [FIX 4] Feature Padding (데이터 모양 맞추기)
        # 범주형 데이터 패딩
        current_cat_feats = X_test_str.shape[1]
        if current_cat_feats < expected_cat_feats:
            diff = expected_cat_feats - current_cat_feats
            # 부족한 컬럼만큼 0으로 채움
            padding = np.zeros((X_test_str.shape[0], diff), dtype=X_test_str.dtype)
            X_test_str = np.hstack([X_test_str, padding])
            logger.warning(f"⚠️ Padded Categorical Features: {current_cat_feats} -> {expected_cat_feats} (Filled with 0)")
        elif current_cat_feats > expected_cat_feats:
            # 넘치는 컬럼은 자름
            X_test_str = X_test_str[:, :expected_cat_feats]
            logger.warning(f"✂️ Truncated Categorical Features: {current_cat_feats} -> {expected_cat_feats}")

        # 수치형 데이터 패딩
        current_num_feats = X_test_num.shape[1]
        if current_num_feats < expected_num_feats:
            diff = expected_num_feats - current_num_feats
            padding = np.zeros((X_test_num.shape[0], diff), dtype=X_test_num.dtype)
            X_test_num = np.hstack([X_test_num, padding])
            logger.warning(f"⚠️ Padded Numerical Features: {current_num_feats} -> {expected_num_feats} (Filled with 0)")
        elif current_num_feats > expected_num_feats:
            X_test_num = X_test_num[:, :expected_num_feats]
            logger.warning(f"✂️ Truncated Numerical Features: {current_num_feats} -> {expected_num_feats}")

        # [FIX 3] Data Clamping (Vocab Size 초과 방지)
        # 패딩 후 수행해야 인덱스 에러가 안 남
        X_clamped = X_test_str.copy()
        for col_idx in range(X_clamped.shape[1]):
            if col_idx in ckpt_cat_max_dict:
                max_val = ckpt_cat_max_dict[col_idx] - 1
                mask = X_clamped[:, col_idx] > max_val
                if np.sum(mask) > 0:
                    X_clamped[mask, col_idx] = 0 # 0번(보통 Unknown)으로 대체
        X_test_str = X_clamped

        # Dataset 정의
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
        
        # 3. 모델 초기화
        encoder = Encoder(
            cnt_cat_features=expected_cat_feats,
            cnt_num_features=expected_num_feats,
            cat_max_dict=ckpt_cat_max_dict,
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
        
        # 4. 가중치 로드
        classifier.load_state_dict(state_dict, strict=False)
        logger.info("✅ Model weights loaded successfully")
        
        classifier.to(device)
        classifier.eval()
        
        # 5. 평가
        metrics = eval_model_func(classifier, test_loader, device=device)
        logger.info(f"Model evaluation metrics: {metrics}")
        
        # DB 저장
        mongodb_client.connect()
        try:
            metrics_id = mongodb_client.store_performance_metrics(
                model_id=model_info['model_id'],
                metrics=metrics,
                dataset_name="test"
            )
            logger.info(f"Metrics stored with ID: {metrics_id}")
            
            # ⭐ [FIX] Do not return 'metadata' containing LabelEncoders
            # Only return essential info needed for downstream tasks
            return {
                "model_id": model_info['model_id'],
                "metrics": metrics,
                # "metadata": metadata,  <-- REMOVED: Contains non-serializable objects
                "data_path": data_path
            }
            
        finally:
            mongodb_client.disconnect()
            
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        import traceback
        logger.error(traceback.format_exc())
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
        data_path = "./data/raw/synthetic_data.csv"
        
        # ⭐ [FIX 1] 10개의 반환값을 하나의 변수로 받음 (Unpacking Error 해결)
        # load_data_for_training returns 10 values:
        # 0:X_train_str, 1:X_train_num, 2:y_train, 3:X_fine_str, 4:X_fine_num, 5:y_fine,
        # 6:X_test_str, 7:X_test_num, 8:y_test, 9:metadata
        results = load_data_for_training(
            data_path=data_path,
            test_size=0.3,
            random_state=42
        )
        
        # 필요한 데이터만 인덱스로 추출
        y_fine = results[5]  # Fine-tuning용 라벨 (Retraining Data)
        y_test = results[8]  # Test용 라벨 (Re-evaluation Data)
        metadata = results[9] # 원본 메타데이터
        
        logger.info(f"Retraining data: {len(y_fine)} samples, Re-evaluation: {len(y_test)} samples")
        
        # ⭐ [FIX 2] JSON 직렬화 가능한 데이터만 골라서 새 딕셔너리 생성
        # LabelEncoder, Scaler 객체는 XCom에 저장할 수 없으므로 제외합니다.
        metadata_lite = {
            'num_categorical_features': metadata['num_categorical_features'],
            'num_numerical_features': metadata['num_numerical_features'],
            'cat_max_dict': metadata['cat_max_dict'],
            'categorical_columns': metadata['categorical_columns'],
            'numerical_columns': metadata['numerical_columns']
        }
        
        # Store data info in XCom
        return {
            "retrain_samples": len(y_fine),
            "reval_samples": len(y_test),
            "metadata": metadata_lite,  # ⭐ 가벼운 메타데이터 전달
            "data_path": data_path
        }
        
    except Exception as e:
        logger.error(f"Error preparing retraining data: {e}")
        raise

def finetune_model(**context):
    """Fine-tune the classifier using code from corp_default_modeling_f.py."""
    import torch
    import numpy as np
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from sklearn.utils.class_weight import compute_class_weight

    logger.info("Starting model fine-tuning using standardized fine-tuning function")
    
    ti = context['ti']
    model_info = ti.xcom_pull(task_ids='load_latest_model')
    retrain_info = ti.xcom_pull(task_ids='prepare_retraining_data')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        # 1. 체크포인트 로드 (구조 파악용)
        model_path = model_info['model_path']
        logger.info(f"Loading checkpoint from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("numeric_embeddings", "num_embeddings")
            new_state_dict[new_key] = v
        state_dict = new_state_dict

        ckpt_cat_max_dict = {}
        max_num_idx = -1
        for k, v in state_dict.items():
            if "encoder.embeddings." in k and ".weight" in k:
                try: idx = int(k.split('.')[2]); ckpt_cat_max_dict[idx] = v.shape[0]
                except: pass
            if "encoder.num_embeddings." in k:
                try: idx = int(k.split('.')[2]); max_num_idx = max(max_num_idx, idx)
                except: pass
        
        expected_cat_feats = max(ckpt_cat_max_dict.keys()) + 1 if ckpt_cat_max_dict else 0
        expected_num_feats = max_num_idx + 1

        # 2. 데이터 로드
        data_path = retrain_info['data_path']
        metadata = retrain_info['metadata']
        results = load_data_for_training(data_path=data_path, test_size=0.3, random_state=42)
        
        X_fine_str = results[3]
        X_fine_num = results[4]
        y_fine = results[5]
        X_test_str = results[6]
        X_test_num = results[7]
        y_test = results[8]

        # Padding
        if X_fine_str.shape[1] < expected_cat_feats:
            padding = np.zeros((X_fine_str.shape[0], expected_cat_feats - X_fine_str.shape[1]), dtype=X_fine_str.dtype)
            X_fine_str = np.hstack([X_fine_str, padding])
            padding_test = np.zeros((X_test_str.shape[0], expected_cat_feats - X_test_str.shape[1]), dtype=X_test_str.dtype)
            X_test_str = np.hstack([X_test_str, padding_test])
        elif X_fine_str.shape[1] > expected_cat_feats:
            X_fine_str = X_fine_str[:, :expected_cat_feats]
            X_test_str = X_test_str[:, :expected_cat_feats]

        if X_fine_num.shape[1] < expected_num_feats:
            padding = np.zeros((X_fine_num.shape[0], expected_num_feats - X_fine_num.shape[1]), dtype=X_fine_num.dtype)
            X_fine_num = np.hstack([X_fine_num, padding])
            padding_test = np.zeros((X_test_num.shape[0], expected_num_feats - X_test_num.shape[1]), dtype=X_test_num.dtype)
            X_test_num = np.hstack([X_test_num, padding_test])
        elif X_fine_num.shape[1] > expected_num_feats:
            X_fine_num = X_fine_num[:, :expected_num_feats]
            X_test_num = X_test_num[:, :expected_num_feats]

        # Clamping
        def clamp_data(X, max_dict):
            X_c = X.copy()
            for col_idx in range(X_c.shape[1]):
                if col_idx in max_dict:
                    max_val = max_dict[col_idx] - 1
                    mask = X_c[:, col_idx] > max_val
                    if np.sum(mask) > 0: X_c[mask, col_idx] = 0 
            return X_c

        X_fine_str = clamp_data(X_fine_str, ckpt_cat_max_dict)
        X_test_str = clamp_data(X_test_str, ckpt_cat_max_dict)

        # 3. 모델 초기화 & 로드
        encoder = Encoder(
            cnt_cat_features=expected_cat_feats,
            cnt_num_features=expected_num_feats,
            cat_max_dict=ckpt_cat_max_dict,
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
        classifier.load_state_dict(state_dict, strict=False)
        classifier.to(device)
        
        # Loader
        class DataLoading(Dataset):
            def __init__(self, X_str, X_num, y):
                self.X_str = X_str; self.X_num = X_num; self.y = y
            def __len__(self): return len(self.y)
            def __getitem__(self, idx):
                return {"str": torch.tensor(self.X_str[idx], dtype=torch.long), "num": torch.tensor(self.X_num[idx], dtype=torch.float), "label": torch.tensor(self.y[idx], dtype=torch.long)}
        
        fine_dataset = DataLoading(X_fine_str, X_fine_num, y_fine)
        test_dataset = DataLoading(X_test_str, X_test_num, y_test)
        fine_loader = DataLoader(fine_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        # ⭐ [FIX] Class Weights 계산 로직 수정 (단일 클래스 문제 해결)
        classes = np.unique(y_fine)
        logger.info(f"Unique classes in fine-tuning data: {classes}")
        
        if len(classes) > 0:
            computed_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_fine)
            # Binary Classification을 가정하고 크기 2짜리 텐서 생성 (기본값 1.0)
            final_class_weights = torch.ones(2, dtype=torch.float).to(device)
            
            # 존재하는 클래스 위치에만 계산된 가중치 할당
            for cls, w in zip(classes, computed_weights):
                if int(cls) < 2: # 0 또는 1 인덱스만 허용
                    final_class_weights[int(cls)] = float(w)
        else:
            # 데이터가 비어있는 경우 (거의 없겠지만)
            final_class_weights = torch.ones(2, dtype=torch.float).to(device)
            
        logger.info(f"Final Class Weights: {final_class_weights}")
        
        optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss(weight=final_class_weights)
        
        # 5. Run Fine-tuning
        training_metrics, test_metrics = finetune_func(
            model=classifier,
            train_loader=fine_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=3
        )
        
        # Save
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        finetuned_path = os.path.join(settings.model_save_path, f"classifier_finetuned_{timestamp}.pth")
        os.makedirs(settings.model_save_path, exist_ok=True)
        torch.save({
            'model_state_dict': classifier.state_dict(),
            'finetuned_from': model_info['model_id'],
            'finetune_timestamp': timestamp
        }, finetuned_path)
        
        logger.info(f"Fine-tuned model saved to {finetuned_path}")
        
        # DB Store
        mongodb_client.connect()
        try:
            finetuned_model_id = mongodb_client.store_model_metadata(
                model_name="default_prediction_classifier",
                model_path=finetuned_path,
                model_version=timestamp,
                hyperparameters={"finetuned_from": model_info['model_id'], "finetune_epochs": 3},
                metrics=test_metrics
            )
            return {
                "model_id": finetuned_model_id,
                "model_path": finetuned_path,
                "finetuned_from": model_info['model_id'],
                "data_path": data_path,
                "test_metrics": test_metrics
            }
        finally:
            mongodb_client.disconnect()
            
    except Exception as e:
        logger.error(f"Error during fine-tuning: {e}")
        import traceback
        logger.error(traceback.format_exc())
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
    )
    
    # Task 2: Evaluate model
    evaluate_task = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model,        
    )
    
    # Task 3: Check performance (branching)
    check_task = BranchPythonOperator(
        task_id='check_model_performance',
        python_callable=check_model_performance,       
    )
    
    # Task 4a: Prepare retraining data (if performance is low)
    prepare_retrain_task = PythonOperator(
        task_id='prepare_retraining_data',
        python_callable=prepare_retraining_data,        
    )
    
    # Task 4b: Fine-tune model
    finetune_task = PythonOperator(
        task_id='finetune_model',
        python_callable=finetune_model,        
    )
    
    # Task 4c: Evaluate fine-tuned model (returns branch decision)
    evaluate_finetuned_task = BranchPythonOperator(
        task_id='evaluate_finetuned_model',
        python_callable=evaluate_finetuned_model,
    )
    
    # Task 5a: Send alert (if max retries exceeded)
    send_alert_task = PythonOperator(
        task_id='send_alert',
        python_callable=send_alert,        
    )
    
    # Task 5b: Send results (both branches converge here)
    send_results_task = PythonOperator(
        task_id='send_evaluation_results',
        python_callable=send_evaluation_results,        
        trigger_rule='none_failed_min_one_success',  # Run if at least one upstream task succeeds
    )
    
   
    # Define task dependencies
    # Main flow
    load_model_task >> evaluate_task >> check_task
    
    # Branch 1: Retraining path (if F1 < threshold)
    check_task >> prepare_retrain_task >> finetune_task >> evaluate_finetuned_task
    
    # Branch 2: Direct to results (if F1 >= threshold)
    check_task >> send_results_task
    
    # Branch 3: Alert path (if max retries exceeded)
    check_task >> send_alert_task
    
    # Finetuned model evaluation branches
    evaluate_finetuned_task >> send_results_task
    evaluate_finetuned_task >> send_alert_task