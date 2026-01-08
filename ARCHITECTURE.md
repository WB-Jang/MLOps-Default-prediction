# Architecture Documentation

## System Architecture

The MLOps Default Prediction pipeline follows a microservices architecture with event-driven data processing.

## High-Level Architecture

```
┌─────────────────┐
│  Data Sources   │
│  (Loan Apps)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│  Kafka Producer │─────▶│  Raw Data    │
│                 │      │  Topic       │
└─────────────────┘      └──────┬───────┘
                                │
                                ▼
                         ┌──────────────┐
                         │  Airflow     │
                         │  Data        │
                         │  Ingestion   │
                         │  DAG         │
                         └──────┬───────┘
                                │
                                ▼
                         ┌──────────────┐
                         │  Processed   │
                         │  Data Topic  │
                         └──────┬───────┘
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
         ┌──────────┐    ┌──────────┐   ┌──────────┐
         │ Training │    │   Model  │   │  MongoDB │
         │   DAG    │    │ Serving  │   │ Storage  │
         └──────────┘    └──────────┘   └──────────┘
```

## Components Detail

### 1. Kafka Messaging Layer

**Purpose**: Decouple data producers from consumers, enable real-time streaming

**Topics**:
- `raw_data`: Unprocessed loan application data
- `processed_data`: Cleaned, normalized data ready for ML
- `commands`: Control messages and events

**Advantages**:
- Asynchronous processing
- Fault tolerance through replication
- Scalable message throughput
- Event sourcing capability

### 2. MongoDB Data Store

**Purpose**: Persistent storage for model artifacts, metadata, and predictions

**Collections**:
- `model_metadata`: Model versions, hyperparameters, file paths
- `performance_metrics`: Evaluation results per model/dataset
- `predictions`: Historical predictions with input/output
- `training_data`: Training job metadata
- `raw_data_logs`: Ingestion tracking

**Schema Design**:
```javascript
// model_metadata
{
  _id: ObjectId,
  model_name: String,
  model_path: String,
  model_version: String,
  model_format: "pth",
  hyperparameters: Object,
  metrics: Object,
  status: String,
  created_at: Date,
  updated_at: Date
}

// predictions
{
  _id: ObjectId,
  model_id: String,
  input_data: Object,
  prediction: Any,
  confidence: Float,
  timestamp: Date
}
```

### 3. Airflow Orchestration

**Purpose**: Schedule and orchestrate ML pipelines

**DAGs**:

1. **data_ingestion_pipeline**
   - Schedule: Hourly
   - Tasks:
     - Ingest raw data
     - Process and validate
     - Send to processed_data topic

2. **model_training_pipeline**
   - Schedule: Daily
   - Tasks:
     - Prepare training data
     - Pretrain encoder (contrastive learning)
     - Train classifier
     - Store model in MongoDB
     - Notify via Kafka

3. **model_evaluation_pipeline**
   - Schedule: Daily (after training)
   - Tasks:
     - Load latest model
     - Evaluate on test set
     - Check performance thresholds
     - Update model status
     - Send results via Kafka

### 4. ML Model Architecture

**Encoder**: Transformer-based tabular encoder
- Categorical features → Embedding layers
- Numerical features → Linear projection
- Multi-head self-attention
- Feed-forward networks

**Classifier**: Binary classification head
- Pooled encoder output
- Fully connected layers
- Dropout regularization
- Binary cross-entropy loss

**Pretraining**: Contrastive learning
- Data augmentation (random masking)
- NT-Xent loss
- Projection head for contrastive space
- Self-supervised on unlabeled data

## Data Flow

### Training Pipeline
```
1. Raw Data → Kafka raw_data topic
2. Airflow ingestion DAG → Processes data → Kafka processed_data topic
3. Airflow training DAG:
   - Consumes from processed_data
   - Pretrains encoder (contrastive)
   - Trains classifier (supervised)
   - Saves model as .pth
   - Stores metadata in MongoDB
   - Sends completion command to Kafka
4. Airflow evaluation DAG:
   - Loads model from MongoDB
   - Evaluates on test data
   - Stores metrics in MongoDB
   - Updates model status
```

### Inference Pipeline
```
1. Inference request received
2. Load latest model from MongoDB
3. Preprocess input data
4. Model prediction
5. Store prediction in MongoDB
6. Return result
```

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| ML Framework | PyTorch | Model development and training |
| Streaming | Apache Kafka | Event streaming and messaging |
| Database | MongoDB | Document storage for models and logs |
| Orchestration | Apache Airflow | Workflow scheduling and monitoring |
| Container | Docker | Application containerization |
| Language | Python 3.10+ | Primary development language |

## Deployment Architecture

### Docker Compose Setup
```
Services:
- zookeeper (Kafka dependency)
- kafka (Message broker)
- mongodb (Database)
- postgres (Airflow metadata DB)
- airflow-webserver
- airflow-scheduler
- airflow-init (one-time setup)
```

### Network Configuration
All services communicate via `mlops-network` bridge network.

### Volume Management
- `mongodb_data`: Persistent MongoDB storage
- `postgres_data`: Persistent Airflow metadata
- `airflow_logs`: Airflow execution logs
- `./models`: Shared model storage

## Scalability Considerations

### Horizontal Scaling
- **Kafka**: Add more brokers, increase partition count
- **Airflow**: Use CeleryExecutor with multiple workers
- **MongoDB**: Implement sharding for large datasets

### Vertical Scaling
- Increase memory for model training containers
- Use GPU-enabled containers for faster training
- Allocate more CPU for Kafka throughput

## Security

### Authentication
- MongoDB: Username/password authentication
- Airflow: Basic auth (upgrade to OAuth for production)
- Kafka: SASL/SSL for production deployments

### Network Security
- Internal network for service communication
- Expose only necessary ports externally
- Use TLS for external connections

### Data Security
- Encrypt MongoDB data at rest
- Secure Kafka topics with ACLs
- Rotate credentials regularly

## Monitoring and Observability

### Recommended Tools
- **Prometheus**: Metrics collection
- **Grafana**: Visualization dashboards
- **Logstash/ELK**: Log aggregation
- **Kafka Manager**: Kafka cluster monitoring
- **Airflow UI**: Pipeline monitoring

### Key Metrics to Track
- Kafka: Message throughput, lag, consumer offset
- MongoDB: Query performance, storage size
- Models: Accuracy, latency, prediction distribution
- Airflow: DAG success rate, task duration

## Future Enhancements

1. **Model Registry**: Integrate MLflow for better model versioning
2. **Feature Store**: Add feature management system
3. **A/B Testing**: Implement model comparison framework
4. **Auto-scaling**: Add Kubernetes for dynamic scaling
5. **Model Serving**: Deploy FastAPI endpoint for predictions
6. **Data Validation**: Integrate Great Expectations
7. **CI/CD**: Add GitHub Actions for automated testing/deployment
