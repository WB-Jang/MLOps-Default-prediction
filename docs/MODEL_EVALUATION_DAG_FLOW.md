# Model Evaluation DAG Flow

## Overview
The model evaluation DAG now supports conditional retraining based on F1 score threshold.

## DAG Flow Diagram

```
┌─────────────────────┐
│ load_latest_model   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  evaluate_model     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────┐
│ check_model_performance │ (BranchPythonOperator)
│   F1 < 0.75?            │
└──────────┬──────────────┘
           │
           ├───────────────────────────┐
           │                           │
           │ (F1 < 0.75)              │ (F1 >= 0.75)
           │                           │
           ▼                           │
┌──────────────────────────┐          │
│ prepare_retraining_data  │          │
└──────────┬───────────────┘          │
           │                           │
           ▼                           │
┌──────────────────────────┐          │
│    finetune_model        │          │
│ (Freeze encoder,         │          │
│  Train classifier only)  │          │
└──────────┬───────────────┘          │
           │                           │
           ▼                           │
┌──────────────────────────┐          │
│ evaluate_finetuned_model │          │
└──────────┬───────────────┘          │
           │                           │
           └───────────┬───────────────┘
                       │
                       ▼
            ┌─────────────────────────┐
            │ send_evaluation_results │
            └─────────────────────────┘
```

## Task Descriptions

### 1. load_latest_model
- Loads the latest classifier model metadata from MongoDB
- Returns model_id, model_path, and model_version

### 2. evaluate_model
- Evaluates the loaded model on test data
- Calculates metrics: accuracy, precision, recall, F1 score, ROC AUC
- Stores metrics in MongoDB

### 3. check_model_performance (BranchPythonOperator)
- Checks if F1 score meets threshold (0.75)
- **Branches to:**
  - `prepare_retraining_data` if F1 < 0.75 (needs retraining)
  - `send_evaluation_results` if F1 >= 0.75 (performance acceptable)
- Updates model status in MongoDB

### 4a. prepare_retraining_data
- Loads evaluation data and splits into retrain/revalidation sets
- Uses 70% of test data for fine-tuning, 30% for re-evaluation
- Returns metadata about prepared data

### 4b. finetune_model
- Loads existing model from checkpoint
- **Freezes encoder parameters** (encoder.parameters().requires_grad = False)
- **Fine-tunes classifier only** with smaller learning rate (1e-4)
- Trains for 5 epochs (fewer than full training)
- Saves fine-tuned model with new timestamp
- Stores fine-tuned model metadata in MongoDB

### 4c. evaluate_finetuned_model
- Evaluates fine-tuned model on re-evaluation data (30% of test set)
- Calculates same metrics as initial evaluation
- Stores re-evaluation metrics in MongoDB
- Returns metrics and model info

### 5. send_evaluation_results
- Converges both branches (retraining and direct paths)
- Stores comprehensive evaluation event in MongoDB
- Includes:
  - Final status (approved/retrained)
  - Final metrics
  - Retraining information if applicable:
    - Original vs fine-tuned F1 scores
    - Performance improvement
- Uses trigger_rule='none_failed_min_one_success' to run if either branch succeeds

## Key Implementation Details

### Conditional Branching
- Uses `BranchPythonOperator` for check_model_performance
- Returns task_id string to indicate next task
- XCom used to pass decision metadata between tasks

### Fine-tuning Strategy
- **Encoder Frozen**: All encoder parameters have requires_grad=False
- **Classifier Only**: Only fc (classification head) parameters are trained
- **Reduced Learning Rate**: 1e-4 instead of 3e-4 for full training
- **Fewer Epochs**: 5 epochs instead of 10 for full training

### Data Strategy
- Retraining uses test data split (70/30 for finetune/reeval)
- This simulates using fresh evaluation data for improvement
- Original training data remains untouched

### MongoDB Storage
- Model metadata with fine-tuning info (finetuned_from, finetune_epochs, etc.)
- Performance metrics for both original and fine-tuned models
- Evaluation events with retraining decision and improvement metrics

## Configuration

### F1 Threshold
Currently set to **0.75** in `check_model_performance` function.
Can be adjusted as needed.

### Fine-tuning Hyperparameters
- Learning rate: 1e-4
- Epochs: 5
- Batch size: 32
- Optimizer: Adam

## Error Handling
All functions include try-except blocks with:
- Detailed error logging via loguru
- MongoDB connection cleanup in finally blocks
- Proper exception propagation for Airflow retry mechanism
