# Implementation Changes Summary

## Overview
This document summarizes the changes made to align the MLOps pipeline with the specified workflow requirements.

## Problem Statement (Korean)
레파지토리의 큰 흐름:
1. 이미 pre-training과 fine-tuning이 완료된 모델을 각각 1개씩 가지고 있어. 이 모델들은 MongoDB에 저장되어서 활용될 거야.
2. 실제 데이터의 분포를 학습한 Distribution model을 활용하여서, 실제와 유사한 Synthetic data를 일별로 생성하고, MongoDB에 저장해
3. Synthetic data들을 processing하고, 이 processing 완료된 data들도 MongoDB에 저장해
4. 저장된 데이터들을 활용하여, fine-tuning 학습까지 완료된 모델을 테스트하고, f1 score가 threshold를 넘지 못하는 경우, 평가데이터를 활용해서 fine-tuning만 진행하는 재학습을 진행해줘. 이때, 평가하는 코드와 fine-tuning하는 코드는, src/model/corp_default_modeling_f.py에 적힌 코드를 그대로 가지고 와서 사용해줘.
5. 재학습 이후에도 f1 score가 threshold를 넘지 못하면 3회 반복하고, alert를 띄워줘
6. 재학습까지 완료된 모델도 새로운 버전으로 기입하여 MongoDB에 저장해줘.
7. 저장된 모델을 배포해
8. 위의 파이프라인(데이터 생성 -> 저장 -> 정제 -> 저장 -> 모델 평가 -> 필요시 재학습 -> 모델 저장)->배포의 과정을 일별로 반복하는 프로젝트야.

## Files Changed

### 1. New Files Created

#### `src/models/evaluation.py`
**Purpose**: Extract evaluation and fine-tuning logic from `corp_default_modeling_f.py` for reuse

**Key Functions**:
- `evaluate_model(model, test_loader, device)`: Evaluates model and returns metrics (accuracy, roc_auc, f1_score)
- `finetune_model(model, train_loader, test_loader, criterion, optimizer, device, epochs)`: Fine-tunes model using the exact training loop from corp_default_modeling_f.py

**Implementation Details**:
```python
def evaluate_model(model, test_loader, device='cpu') -> Dict[str, float]:
    """Uses evaluation code from corp_default_modeling_f.py"""
    # - Model evaluation on test data
    # - Returns accuracy, roc_auc, f1_score
    
def finetune_model(model, train_loader, test_loader, criterion, 
                   optimizer, device, epochs=3):
    """Uses fine-tuning code from corp_default_modeling_f.py"""
    # - Training loop with loss tracking
    # - Per-epoch metrics logging
    # - Final test evaluation
    # - Returns training_metrics and test_metrics
```

#### `src/airflow/dags/model_deployment_dag.py`
**Purpose**: Implement model deployment step (#7 in workflow)

**Key Tasks**:
1. `wait_for_evaluation`: External task sensor waiting for evaluation completion
2. `wait_for_model_ready`: Check for approved model in MongoDB
3. `deploy_model`: Copy model to production deployment path
4. `notify_deployment_complete`: Store deployment event in MongoDB

**Implementation Details**:
- Uses ExternalTaskSensor to coordinate with evaluation DAG
- Deploys to `./models/deployed/current_model.pth`
- Tracks deployments in MongoDB `deployment_events` collection
- Creates version file for deployment tracking

#### `WORKFLOW.md`
**Purpose**: Comprehensive Korean documentation of the complete workflow

**Contents**:
- Complete workflow diagram with retry mechanism
- Detailed implementation of each step (1-8)
- Code snippets showing how corp_default_modeling_f.py code is used
- MongoDB collection structure
- Retry and alert mechanism details
- Configuration settings
- Monitoring queries
- Troubleshooting guide

### 2. Modified Files

#### `src/airflow/dags/data_ingestion_dag.py`
**Changes**:
1. **Data Generation** (#2 in workflow):
   - Now uses `distribution_model.pkl` to generate synthetic data
   - Added fallback mechanism if distribution model not found
   - Generates data with realistic distribution

2. **Data Processing** (#3 in workflow):
   - Complete preprocessing pipeline:
     - Load raw data
     - Handle null values
     - Convert data types
     - Date processing
     - Data shrinkage
     - Encoding and scaling
   - **Store processed data to MongoDB** `processed_data` collection
   - Store processing metadata to `processing_metadata` collection

**Key Code**:
```python
def generate_data_task(**context):
    # Use distribution model
    processor.data_generation(data_quantity=10000)
    
def preprocess_data_task(**context):
    # Process data
    scaled_df, str_cols, num_cols, nunique = processor.detecting_type_encoding()
    
    # Store to MongoDB
    processed_data_records = scaled_df.to_dict('records')
    collection.insert_many(processed_data_records)
```

#### `src/airflow/dags/model_evaluation_dag.py`
**Major Overhaul** - Implements requirements #4, #5, #6:

**Changes**:
1. **Import evaluation functions**:
   ```python
   from src.models.evaluation import evaluate_model as eval_model_func
   from src.models.evaluation import finetune_model as finetune_func
   ```

2. **Enhanced evaluate_model** (#4):
   - Now uses actual data loading
   - Creates DataLoader for test data
   - Uses `eval_model_func` from corp_default_modeling_f.py
   - Returns actual metrics

3. **Enhanced check_model_performance** (#5):
   - Tracks retry count via XCom
   - Implements max retry logic (MAX_RETRAIN_ATTEMPTS = 3)
   - Branches to:
     - `prepare_retraining_data` if F1 < threshold and retries < 3
     - `send_alert` if retries >= 3
     - `send_evaluation_results` if F1 >= threshold

4. **Enhanced finetune_model** (#4, #6):
   - Loads actual data for fine-tuning
   - Uses evaluation data (X_fine_str, X_fine_num, y_fine)
   - Computes class weights for imbalanced data
   - Uses `finetune_func` from corp_default_modeling_f.py
   - Saves model with new version timestamp
   - Stores in MongoDB with metadata

5. **New send_alert function** (#5):
   - Creates alert document with:
     - Alert type: MODEL_PERFORMANCE_FAILURE
     - Severity: CRITICAL
     - Model ID, F1 score, retry attempts
     - Detailed message
   - Stores in MongoDB `alerts` collection
   - Stores in `evaluation_events` for tracking

6. **Modified evaluate_finetuned_model**:
   - Now uses BranchPythonOperator
   - Checks if retry needed
   - Returns next task:
     - `check_model_performance` to loop back for retry
     - `send_alert` if max retries reached
     - `send_evaluation_results` if performance good

7. **Updated DAG task dependencies**:
   ```python
   # Loop-back for retry
   evaluate_finetuned_task >> check_task  # Enables retry
   
   # Alert path
   check_task >> send_alert_task
   evaluate_finetuned_task >> send_alert_task
   ```

**Key Constants**:
```python
F1_THRESHOLD = 0.75
MAX_RETRAIN_ATTEMPTS = 3
```

#### `src/models/__init__.py`
**Changes**:
- Added exports for evaluation functions:
  ```python
  from .evaluation import (
      evaluate_model,
      finetune_model
  )
  ```

#### `README.md`
**Changes**:
1. **Updated Project Flow section**:
   - Added complete daily pipeline diagram (1-8)
   - Added enhanced DAG flow with retry mechanism
   - Listed key features (retry, alert, code reuse, versioning)

2. **Updated Project Structure**:
   - Added `evaluation.py`
   - Added `distribution_model.pkl`
   - Added `model_deployment_dag.py`
   - Added `WORKFLOW.md`

3. **Added Advanced Usage section**:
   - Link to WORKFLOW.md
   - MongoDB monitoring queries
   - Alert checking queries
   - Configuration adjustments

## Workflow Implementation Details

### Daily Execution Sequence

1. **data_pipeline_dag** runs daily:
   - Generates synthetic data using distribution model
   - Stores raw data to MongoDB
   - Processes data
   - Stores processed data to MongoDB

2. **model_evaluation_pipeline** runs daily:
   - Loads latest fine-tuned model
   - Evaluates using test data
   - If F1 < 0.75:
     - Attempt 1: Fine-tune → Evaluate → Check
     - Attempt 2: Fine-tune → Evaluate → Check (if still low)
     - Attempt 3: Fine-tune → Evaluate → Check (if still low)
     - If still low after 3 attempts: Send Alert
   - If F1 >= 0.75: Continue to deployment
   - All retrained models saved with new versions

3. **model_deployment_pipeline** runs daily:
   - Waits for evaluation completion
   - Checks for approved model
   - Deploys to production
   - Stores deployment event

### Retry Mechanism Implementation

**XCom State Management**:
```python
# check_model_performance
retry_count = ti.xcom_pull(key='retry_count', default=0)
if needs_retraining and retry_count < MAX_RETRAIN_ATTEMPTS:
    ti.xcom_push(key='retry_count', value=retry_count + 1)
    return 'prepare_retraining_data'
elif retry_count >= MAX_RETRAIN_ATTEMPTS:
    return 'send_alert'
```

**Loop-back Logic**:
```python
# evaluate_finetuned_model (BranchPythonOperator)
if test_metrics['f1_score'] < F1_THRESHOLD:
    if retry_count < MAX_RETRAIN_ATTEMPTS:
        return 'check_model_performance'  # Loop back
    else:
        return 'send_alert'
else:
    return 'send_evaluation_results'
```

### Code Reuse from corp_default_modeling_f.py

**Evaluation**:
- Original location: Lines 442-457 in corp_default_modeling_f.py
- New location: `evaluate_model()` in src/models/evaluation.py
- Usage: model_evaluation_dag.py line ~62-100

**Fine-tuning**:
- Original location: Lines 408-469 in corp_default_modeling_f.py
- New location: `finetune_model()` in src/models/evaluation.py
- Usage: model_evaluation_dag.py line ~188-290
- Includes:
  - Training loop with class weights
  - Per-epoch metric logging
  - Test evaluation
  - Format matches original exactly

## MongoDB Collections Used

### New/Enhanced Collections:
1. **processed_data**: Stores processed/scaled data
2. **processing_metadata**: Tracks processing runs
3. **alerts**: Critical alerts for failed retraining
4. **deployment_events**: Deployment history
5. **evaluation_events**: Enhanced with retry information

### Collection Schemas:

**alerts**:
```javascript
{
  alert_type: "MODEL_PERFORMANCE_FAILURE",
  severity: "CRITICAL",
  timestamp: ISODate,
  model_id: ObjectId,
  f1_score: Number,
  f1_threshold: Number,
  retry_attempts: Number,
  message: String
}
```

**processed_data**:
```javascript
{
  ...feature_columns...,
  processing_timestamp: ISODate,
  dataset_name: String
}
```

**deployment_events**:
```javascript
{
  event_type: "model_deployed",
  timestamp: ISODate,
  model_id: ObjectId,
  model_version: String,
  deployment_path: String,
  status: String
}
```

## Testing Recommendations

### Unit Tests:
1. Test `evaluate_model()` function
2. Test `finetune_model()` function
3. Test retry counter logic
4. Test alert generation

### Integration Tests:
1. Run data_pipeline_dag end-to-end
2. Run model_evaluation_pipeline with mock low F1 score
3. Verify retry mechanism (should retry 3 times)
4. Verify alert creation after max retries
5. Run deployment_dag after successful evaluation

### Manual Testing:
```bash
# 1. Generate test data
python generate_raw_data.py --num-rows 1000

# 2. Test evaluation functions
python -c "
from src.models.evaluation import evaluate_model, finetune_model
# Test code here
"

# 3. Trigger DAGs manually in Airflow UI
# 4. Monitor MongoDB for:
#    - processed_data collection
#    - alerts collection (if F1 low)
#    - deployment_events collection
```

## Configuration

### Adjustable Parameters:

**F1 Threshold**:
```python
# src/airflow/dags/model_evaluation_dag.py
F1_THRESHOLD = 0.75  # Change to adjust sensitivity
```

**Max Retries**:
```python
# src/airflow/dags/model_evaluation_dag.py
MAX_RETRAIN_ATTEMPTS = 3  # Change to allow more/fewer retries
```

**Fine-tuning Epochs**:
```python
# src/airflow/dags/model_evaluation_dag.py
# In finetune_model function
training_metrics, test_metrics = finetune_func(
    ...
    epochs=3  # Change number of epochs
)
```

**Data Generation Quantity**:
```python
# src/airflow/dags/data_ingestion_dag.py
processor.data_generation(data_quantity=10000)  # Adjust size
```

## Benefits of Changes

1. **Code Reusability**: corp_default_modeling_f.py logic now modular and reusable
2. **Automatic Recovery**: Up to 3 retries without manual intervention
3. **Alerting**: Immediate notification of critical failures
4. **Version Control**: All models tracked with timestamps
5. **Complete Automation**: Entire pipeline runs daily without human intervention
6. **Traceability**: All events logged in MongoDB
7. **Deployment Safety**: Only approved models deployed to production

## Future Enhancements

1. **Alert Integration**:
   - Email notifications
   - Slack webhooks
   - PagerDuty integration

2. **Advanced Deployment**:
   - Blue-green deployment
   - Canary releases
   - Rollback capability

3. **Monitoring**:
   - Prometheus metrics
   - Grafana dashboards
   - Model drift detection

4. **Optimization**:
   - Hyperparameter tuning during retries
   - Dynamic threshold adjustment
   - Automated data quality checks
