# MLOps Architecture Documentation

## System Overview

This document describes the architecture of the Loan Default Prediction MLOps system.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     MLOps Pipeline                           │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐                                            │
│  │ Data Sources │                                            │
│  └──────┬───────┘                                            │
│         │                                                     │
│         ▼                                                     │
│  ┌──────────────────────────────────────────────┐           │
│  │          PostgreSQL Database                  │           │
│  │  ┌────────────────────────────────────────┐  │           │
│  │  │  • Raw Data                            │  │           │
│  │  │  • Processed Data                      │  │           │
│  │  │  • Model Metadata                      │  │           │
│  │  │  • Predictions                         │  │           │
│  │  │  • Performance Metrics                 │  │           │
│  │  └────────────────────────────────────────┘  │           │
│  └──────┬───────────────────────────────────────┘           │
│         │                                                     │
│         ▼                                                     │
│  ┌──────────────────────────────────────────────┐           │
│  │         Apache Airflow                        │           │
│  │  ┌────────────────────────────────────────┐  │           │
│  │  │  DAG 1: Data Collection                │  │           │
│  │  │    • Collect new loan data             │  │           │
│  │  │    • Validate data quality             │  │           │
│  │  │    • Store in PostgreSQL               │  │           │
│  │  └────────────────────────────────────────┘  │           │
│  │  ┌────────────────────────────────────────┐  │           │
│  │  │  DAG 2: Model Training                 │  │           │
│  │  │    • Prepare training data             │  │           │
│  │  │    • Pretrain with contrastive learning│  │           │
│  │  │    • Train classifier                  │  │           │
│  │  │    • Save model and metadata           │  │           │
│  │  └────────────────────────────────────────┘  │           │
│  │  ┌────────────────────────────────────────┐  │           │
│  │  │  DAG 3: Validation & Deployment        │  │           │
│  │  │    • Evaluate active model             │  │           │
│  │  │    • Check F1-score threshold (0.8)    │  │           │
│  │  │    • Trigger retraining if needed      │  │           │
│  │  │    • Deploy new model                  │  │           │
│  │  │    • Save performance metrics          │  │           │
│  │  └────────────────────────────────────────┘  │           │
│  └──────┬───────────────────────────────────────┘           │
│         │                                                     │
│         ▼                                                     │
│  ┌──────────────────────────────────────────────┐           │
│  │         FastAPI Application                   │           │
│  │  ┌────────────────────────────────────────┐  │           │
│  │  │  Endpoints:                            │  │           │
│  │  │    • GET  /health                      │  │           │
│  │  │    • POST /predict                     │  │           │
│  │  │    • POST /reload-model                │  │           │
│  │  │    • GET  /model-info                  │  │           │
│  │  └────────────────────────────────────────┘  │           │
│  └───────────────────────────────────────────────┘          │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. PostgreSQL Database

**Purpose**: Central data storage for all system data

**Schema**:
- `loan_data.raw_data`: Stores raw loan application data
- `loan_data.processed_data`: Stores preprocessed features
- `loan_data.model_metadata`: Tracks model versions and metrics
- `loan_data.predictions`: Logs all predictions made
- `loan_data.model_performance`: Historical performance tracking

**Key Features**:
- ACID compliance for data integrity
- Indexed queries for performance
- Automatic timestamp tracking

### 2. Apache Airflow

**Purpose**: Orchestrate and schedule MLOps workflows

#### DAG 1: Data Collection (Schedule: Daily)

```python
collect_data → validate_data
```

- Collects new loan data from sources
- Validates data quality and schema
- Stores validated data in PostgreSQL

#### DAG 2: Model Training (Schedule: Triggered)

```python
prepare_training_data → pretrain_model → train_classifier
```

- Retrieves new data for training
- Pretrains encoder with contrastive learning
- Trains classifier on labeled data
- Saves model with version control

#### DAG 3: Validation & Deployment (Schedule: Daily)

```python
evaluate_active_model → check_performance_threshold
    ├── trigger_retraining (if F1 < 0.8)
    └── skip_retraining (if F1 >= 0.8)
        → save_performance_metrics
```

- Evaluates current model performance
- Checks if F1-score is below 0.8 threshold
- Triggers retraining if performance degrades
- Deploys new models automatically

### 3. Model Architecture

#### TabTransformer

**Encoder**:
```
Input Layer
  ├── Categorical Features → Embedding(n_categories, d_model)
  └── Numerical Features   → Linear(1, d_model)
                              ↓
                      Concatenate (seq_len, d_model)
                              ↓
              Transformer Encoder (6 layers)
                - Multi-head Attention (4 heads)
                - Feed-forward Network (64 dim)
                - Layer Normalization
                - Residual Connections
                              ↓
                         Output (seq_len, d_model)
```

**Classifier Head**:
```
Encoded Features → Mean Pooling
                       ↓
              Linear(d_model, final_hidden)
                       ↓
                    ReLU + Dropout
                       ↓
              Linear(final_hidden, 2)
                       ↓
                   Output (2 classes)
```

**Pretraining Strategy**:
- Contrastive Learning with NT-Xent Loss
- Data augmentation through random masking
- Two augmented views per sample
- Temperature-scaled cosine similarity

### 4. FastAPI Application

**Purpose**: Serve predictions via REST API

**Endpoints**:

1. `GET /health`: Service health check
2. `POST /predict`: Make predictions
3. `POST /reload-model`: Reload active model
4. `GET /model-info`: Get model metadata

**Features**:
- Automatic model loading on startup
- Model version tracking
- Prediction logging to database
- Error handling and validation

## Data Flow

### Training Flow

```
1. New Data Arrives
   ↓
2. Data Collection DAG
   ├── Validate data
   └── Store in raw_data table
   ↓
3. Training DAG (Triggered when samples > threshold)
   ├── Retrieve new data
   ├── Pretrain encoder
   ├── Train classifier
   └── Save model + metadata
   ↓
4. Validation DAG
   ├── Evaluate on test set
   ├── Check F1-score
   └── Deploy if F1 >= 0.8
```

### Inference Flow

```
1. API Request
   ↓
2. Load Active Model (if not cached)
   ↓
3. Preprocess Features
   ↓
4. Run Inference
   ↓
5. Log Prediction
   ↓
6. Return Response
```

### Monitoring Flow

```
1. Validation DAG (Daily)
   ↓
2. Evaluate Active Model
   ├── Calculate metrics
   └── Store in model_performance table
   ↓
3. Check 7-day Average F1-score
   ↓
4. If < 0.8 Threshold
   ├── Trigger Training DAG
   └── Retrain with new data
```

## Deployment Strategy

### Docker Compose Services

1. **postgres**: PostgreSQL database
2. **airflow-webserver**: Airflow UI (port 8080)
3. **airflow-scheduler**: Airflow task scheduler
4. **app**: FastAPI application (port 8000)

### Volume Mounts

- `models/`: Persistent model storage
- `airflow/logs/`: Airflow execution logs
- `postgres_data/`: Database persistence

## Scalability Considerations

### Current Architecture (Single Node)
- Suitable for small to medium workloads
- LocalExecutor for Airflow
- Single API instance

### Future Scaling Options

1. **Horizontal Scaling**:
   - Multiple API instances behind load balancer
   - CeleryExecutor for Airflow
   - Redis/RabbitMQ for task queue

2. **Database Scaling**:
   - Read replicas for queries
   - Partitioning for historical data
   - Connection pooling

3. **Model Serving**:
   - Model caching layer (Redis)
   - Batch prediction endpoint
   - GPU support for inference

## Security Considerations

1. **Database**:
   - Use strong passwords (change defaults)
   - Enable SSL/TLS connections
   - Regular backups

2. **API**:
   - Add authentication/authorization
   - Rate limiting
   - Input validation

3. **Airflow**:
   - Change default admin password
   - Use secrets backend
   - Enable authentication

## Monitoring and Observability

### Metrics to Track

1. **Model Performance**:
   - F1-score, Accuracy, Precision, Recall
   - ROC-AUC
   - Prediction distribution

2. **System Health**:
   - API response time
   - Airflow DAG success rate
   - Database query performance

3. **Business Metrics**:
   - Predictions per day
   - Model drift indicators
   - Data quality metrics

### Recommended Tools

- Prometheus + Grafana for metrics
- ELK Stack for logs
- MLflow for experiment tracking

## Disaster Recovery

### Backup Strategy

1. **Database**: Daily automated backups
2. **Models**: Version-controlled in models/
3. **Configuration**: Version-controlled in Git

### Recovery Procedures

1. Database failure: Restore from backup
2. Model corruption: Revert to previous version
3. Service failure: Docker Compose restart

## Performance Optimization

### Training
- Batch size tuning
- Learning rate scheduling
- Early stopping
- Mixed precision training (GPU)

### Inference
- Model caching
- Batch predictions
- Feature preprocessing optimization
- TorchScript compilation

## Future Enhancements

1. **Model Improvements**:
   - Ensemble models
   - AutoML integration
   - Feature engineering automation

2. **Pipeline Enhancements**:
   - A/B testing framework
   - Shadow deployment
   - Blue-green deployment

3. **Monitoring**:
   - Real-time drift detection
   - Anomaly detection
   - Alert system

4. **Data Management**:
   - Feature store integration
   - Data versioning
   - Data quality monitoring
