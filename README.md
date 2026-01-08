# MLOps Default Prediction Pipeline

A comprehensive MLOps pipeline for loan default prediction using PyTorch, Kafka, Airflow, and MongoDB.

## Architecture Overview

This project implements a complete MLOps pipeline with the following components:

### Core Components

1. **Machine Learning Models**
   - Transformer-based encoder for tabular data
   - Tab Transformer classifier for binary default prediction
   - Contrastive learning pretraining support
   - Models saved in `.pth` format (PyTorch standard)

2. **Kafka Integration**
   - **Raw Data Topic**: Receives incoming loan application data
   - **Processed Data Topic**: Distributes preprocessed data ready for training/inference
   - **Commands Topic**: Handles pipeline control commands and events
   - Producers and consumers for data streaming

3. **MongoDB Database**
   - **Model Metadata**: Stores model versions, hyperparameters, and paths
   - **Performance Metrics**: Tracks model evaluation results
   - **Predictions**: Logs all model predictions with timestamps
   - **Training Logs**: Records training pipeline execution data

4. **Airflow Orchestration**
   - **Data Ingestion DAG**: Automated data collection via Kafka
   - **Model Training DAG**: End-to-end training pipeline
   - **Model Evaluation DAG**: Automated model evaluation and validation

## Project Structure

```
MLOps-Default-prediction/
├── src/
│   ├── models/           # Model architecture and training
│   │   ├── network.py    # Neural network definitions
│   │   └── training.py   # Training utilities
│   ├── data/             # Data processing
│   │   └── preprocessing.py
│   ├── kafka/            # Kafka integration
│   │   ├── producer.py
│   │   └── consumer.py
│   ├── database/         # MongoDB integration
│   │   └── mongodb.py
│   └── airflow/
│       └── dags/         # Airflow DAGs
│           ├── data_ingestion_dag.py
│           ├── model_training_dag.py
│           └── model_evaluation_dag.py
├── config/               # Configuration
│   └── settings.py
├── tests/                # Unit tests
├── docker/               # Docker configurations
│   ├── Dockerfile.airflow
│   └── Dockerfile.app
├── docker-compose.yml    # Full stack deployment
├── requirements.txt
├── train.py             # Standalone training script
└── README.md
```

## Setup Instructions

### Prerequisites

- Docker and Docker Compose
- Python 3.10+
- 8GB+ RAM recommended

### Quick Start with Docker

1. **Clone the repository**
   ```bash
   git clone https://github.com/WB-Jang/MLOps-Default-prediction.git
   cd MLOps-Default-prediction
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start all services**
   ```bash
   docker-compose up -d
   ```

4. **Initialize Airflow (first time only)**
   ```bash
   docker-compose run airflow-init
   ```

5. **Access services**
   - Airflow UI: http://localhost:8080 (admin/admin)
   - Kafka: localhost:9092
   - MongoDB: localhost:27017

### Manual Setup (Development)

1. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup Kafka**
   ```bash
   # Using Docker
   docker run -d --name kafka -p 9092:9092 \
     -e KAFKA_ZOOKEEPER_CONNECT=zookeeper:2181 \
     confluentinc/cp-kafka:7.5.0
   ```

4. **Setup MongoDB**
   ```bash
   docker run -d --name mongodb -p 27017:27017 \
     -e MONGO_INITDB_ROOT_USERNAME=admin \
     -e MONGO_INITDB_ROOT_PASSWORD=changeme \
     mongo:7.0
   ```

5. **Setup Airflow**
   ```bash
   export AIRFLOW_HOME=~/airflow
   airflow db init
   airflow users create --username admin --password admin \
     --firstname Admin --lastname User --role Admin \
     --email admin@example.com
   ```

## Usage

### Training a Model

**Using the standalone script:**
```bash
# Pretrain the encoder
python train.py --pretrain --epochs 10

# Train the classifier
python train.py --train --epochs 20

# Both pretraining and training
python train.py --pretrain --train --epochs 10
```

**Using Airflow:**
1. Navigate to Airflow UI (http://localhost:8080)
2. Enable the `model_training_pipeline` DAG
3. Trigger the DAG manually or let it run on schedule

### Data Ingestion

**Send data to Kafka:**
```python
from src.kafka.producer import kafka_producer

kafka_producer.connect()
kafka_producer.send_raw_data({
    "loan_id": "12345",
    "features": {...}
})
kafka_producer.disconnect()
```

**Using Airflow DAG:**
- The `data_ingestion_pipeline` DAG runs hourly
- Automatically processes data from Kafka topics

### Model Inference

```python
import torch
from src.models import TabTransformerClassifier, load_model
from src.database.mongodb import mongodb_client

# Get latest model from MongoDB
mongodb_client.connect()
model_metadata = mongodb_client.get_model_metadata("default_prediction_classifier")

# Load model
model = TabTransformerClassifier(...)
model = load_model(model, model_metadata['model_path'])

# Make predictions
model.eval()
with torch.no_grad():
    predictions = model(x_cat, x_num)
```

## Kafka Topics

### Raw Data Topic
- **Name**: `raw_data`
- **Purpose**: Receives incoming loan application data
- **Schema**: JSON with loan features

### Processed Data Topic
- **Name**: `processed_data`
- **Purpose**: Distributes preprocessed, training-ready data
- **Schema**: JSON with normalized features

### Commands Topic
- **Name**: `commands`
- **Purpose**: Pipeline control and event notifications
- **Events**: `training_complete`, `evaluation_complete`, etc.

## MongoDB Collections

### model_metadata
Stores model version information and hyperparameters.

### performance_metrics
Records model evaluation metrics on different datasets.

### predictions
Logs all model predictions for monitoring and analysis.

### training_data
Tracks training data batches and preprocessing status.

## Model File Format

All models are saved in PyTorch's `.pth` format:

```python
# Saving
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'metrics': metrics
}, 'model.pth')

# Loading
checkpoint = torch.load('model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

## Testing

Run the test suite:

```bash
# All tests
pytest tests/

# Specific test file
pytest tests/test_kafka.py

# With coverage
pytest --cov=src tests/
```

## Configuration

Edit `config/settings.py` or set environment variables:

```bash
# MongoDB
MONGODB_HOST=localhost
MONGODB_PORT=27017
MONGODB_DATABASE=mlops_default_prediction

# Kafka
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_RAW_DATA_TOPIC=raw_data
KAFKA_PROCESSED_DATA_TOPIC=processed_data
KAFKA_COMMANDS_TOPIC=commands

# Model
MODEL_SAVE_PATH=./models
D_MODEL=32
NHEAD=4
NUM_LAYERS=6
```

## Deployment

### Production Deployment

1. **Update environment variables** for production settings
2. **Scale Kafka brokers** for high throughput
3. **Configure MongoDB replica set** for high availability
4. **Use Kubernetes** for Airflow workers (optional)
5. **Setup monitoring** with Prometheus/Grafana

### Scaling Considerations

- **Kafka**: Add more brokers and partitions for parallel processing
- **MongoDB**: Use sharding for large-scale data
- **Airflow**: Use CeleryExecutor with multiple workers
- **Model Serving**: Deploy model behind API gateway (FastAPI/Flask)

## Troubleshooting

### Kafka Connection Issues
```bash
# Check Kafka is running
docker ps | grep kafka

# View Kafka logs
docker logs kafka
```

### MongoDB Connection Issues
```bash
# Check MongoDB is running
docker ps | grep mongodb

# Test connection
mongo mongodb://admin:changeme@localhost:27017
```

### Airflow DAG Issues
```bash
# Check DAG syntax
python src/airflow/dags/model_training_dag.py

# View Airflow logs
docker logs airflow-scheduler
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Contact

For questions or issues, please open a GitHub issue.
