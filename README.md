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

2. **Data Generation & Processing**
   - Synthetic data generation for loan applications
   - Raw data preprocessing and feature engineering
   - Categorical and numerical feature handling
   - Train/test split with stratification

3. **MongoDB Database**
   - **Model Metadata**: Stores model versions, hyperparameters, and paths
   - **Performance Metrics**: Tracks model evaluation results
   - **Predictions**: Logs all model predictions with timestamps
   - **Training Logs**: Records training pipeline execution data

4. **Airflow Orchestration**
   - **Data Ingestion DAG**: Automated data collection and distribution
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
│   │   ├── preprocessing.py          # Data preprocessing utilities
│   │   ├── data_loader.py            # Data loading utilities
│   │   └── data_augmentation_generator.py  # SDV-based augmentation
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
├── logs/                # Application logs
├── tests/               # Unit tests
├── docker/              # Docker configurations
│   ├── Dockerfile.airflow
│   └── Dockerfile.app
├── docker-compose.yml   # Full stack deployment
├── requirements.txt
├── generate_raw_data.py # Script to generate synthetic data
├── train.py            # Standalone training script
└── README.md
```

## Setup Instructions

### Prerequisites

- Docker and Docker Compose (for full stack deployment)
- Python 3.10+
- 8GB+ RAM recommended

### Quick Start - Complete Pipeline Setup

Follow these steps to set up and run the entire MLOps pipeline:

#### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/WB-Jang/MLOps-Default-prediction.git
cd MLOps-Default-prediction

# Create necessary directories
mkdir -p data/raw models logs

# Configure environment (optional)
cp .env.example .env
# Edit .env with your configuration if needed
```

#### 2. Generate Raw Data

**This is the first step and is required for the pipeline to work.**

```bash
# Generate synthetic loan application data (10,000 rows by default)
python generate_raw_data.py

# Generate with custom parameters
python generate_raw_data.py --num-rows 50000 --output-dir ./data/raw

# Generate without target variable (for inference only)
python generate_raw_data.py --num-rows 10000 --no-target

# View help for all options
python generate_raw_data.py --help
```

**Output:**
- `data/raw/synthetic_data.csv` - Raw loan application data
- `data/raw/synthetic_data_info.txt` - Dataset statistics and information

**Data Schema:**
The generated data includes:
- **Categorical Features (10)**: employment_type, income_category, education_level, credit_score_category, payment_history, loan_purpose, property_type, marital_status, dependents, region
- **Numerical Features (15)**: annual_income, loan_amount, debt_to_income_ratio, credit_score, existing_debt, employment_length_years, months_at_current_job, loan_term_months, interest_rate, num_credit_lines, num_credit_inquiries, revolving_balance, revolving_utilization, age, months_since_last_delinquency
- **Target Variable**: default (0 = no default, 1 = default)

#### 3. Option A: Docker Deployment (Recommended for Full Pipeline)

```bash
# Start all services (Kafka, MongoDB, Airflow)
docker-compose up -d

# Check service status
docker-compose ps

# Initialize Airflow (first time only)
docker-compose run airflow-init

# View logs
docker-compose logs -f airflow-scheduler
docker-compose logs -f kafka
docker-compose logs -f mongodb
```

**Access Services:**
- Airflow UI: http://localhost:8080 (username: `admin`, password: `admin`)
- MongoDB: localhost:27017

**Run the Pipeline via Airflow:**

1. Navigate to Airflow UI (http://localhost:8080)
2. Enable the DAGs:
   - `data_ingestion_pipeline` - Loads raw data and stores to MongoDB
   - `model_training_pipeline` - Trains models end-to-end
   - `model_evaluation_pipeline` - Evaluates trained models
3. Trigger DAGs manually or let them run on schedule:
   - Data Ingestion: Runs hourly
   - Model Training: Runs daily
   - Model Evaluation: Runs daily after training

#### 4. Option B: Manual Setup (Development)

**Install Dependencies:**

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Start Required Services:**

```bash
# Start MongoDB (in separate terminal)
docker run -d --name mongodb -p 27017:27017 \
  -e MONGO_INITDB_ROOT_USERNAME=admin \
  -e MONGO_INITDB_ROOT_PASSWORD=changeme \
  mongo:7.0
```

**Train Models Locally:**

```bash
# Train with generated data (uses data/raw/synthetic_data.csv)
python train.py --train --epochs 10

# Pretrain encoder only
python train.py --pretrain --epochs 10

# Both pretraining and training
python train.py --pretrain --train --epochs 10

# Train with custom data path
python train.py --train --data-path ./data/raw/custom_data.csv --epochs 20

# View all training options
python train.py --help
```

**Training Output:**
- Models saved to `./models/` directory
- `pretrained_encoder_YYYYMMDD_HHMMSS.pth` - Pretrained encoder
- `classifier_YYYYMMDD_HHMMSS.pth` - Trained classifier
- Training logs in `./logs/` directory

## Complete Pipeline Workflow

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. DATA GENERATION                                              │
│    python generate_raw_data.py                                  │
│    └─> data/raw/synthetic_data.csv                             │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. DATA INGESTION                                               │
│    Airflow DAG: data_ingestion_pipeline                         │
│    └─> Loads CSV → Stores to MongoDB raw_data collection       │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. DATA PROCESSING                                              │
│    DAG processes raw data → MongoDB processed_data collection   │
│    └─> Normalizes features, encodes categoricals               │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. MODEL TRAINING                                               │
│    Airflow DAG: model_training_pipeline                         │
│    ├─> Loads data from CSV or MongoDB                          │
│    ├─> Pretrain encoder (contrastive learning)                 │
│    ├─> Train classifier (supervised)                            │
│    ├─> Save models to ./models/*.pth                            │
│    └─> Store metadata in MongoDB                                │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. MODEL EVALUATION                                             │
│    Airflow DAG: model_evaluation_pipeline                       │
│    ├─> Load latest model from MongoDB                           │
│    ├─> Evaluate on test set                                     │
│    └─> Store metrics in MongoDB                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Step-by-Step Execution

**1. Generate Raw Data:**
```bash
python generate_raw_data.py --num-rows 10000
```

**2. Start Services:**
```bash
docker-compose up -d
```

**3. Trigger Pipeline via Airflow UI or CLI:**
```bash
# Using Airflow CLI (if installed)
airflow dags trigger data_ingestion_pipeline
airflow dags trigger model_training_pipeline
airflow dags trigger model_evaluation_pipeline
```

**4. Monitor Pipeline:**
- Check Airflow UI for DAG execution status
- View logs in `./logs/` directory
- Query MongoDB for model metadata and metrics

**5. Use Trained Models:**
```python
import torch
from src.models import TabTransformerClassifier
from src.database.mongodb import mongodb_client

# Connect to MongoDB
mongodb_client.connect()

# Get latest model metadata
model_metadata = mongodb_client.get_model_metadata("default_prediction_classifier")

# Load model
model = TabTransformerClassifier(...)  # Initialize with saved hyperparameters
checkpoint = torch.load(model_metadata['model_path'])
model.load_state_dict(checkpoint['model_state_dict'])

# Make predictions
model.eval()
with torch.no_grad():
    predictions = model(x_cat, x_num)
```

## Configuration

### Environment Variables

Edit `.env` file or set environment variables:

```bash
# MongoDB Configuration
MONGODB_HOST=localhost
MONGODB_PORT=27017
MONGODB_DATABASE=mlops_default_prediction

# Model Configuration
MODEL_SAVE_PATH=./models
D_MODEL=32
NHEAD=4
NUM_LAYERS=6
DIM_FEEDFORWARD=64
DROPOUT_RATE=0.3

# Training Configuration
BATCH_SIZE=32
LEARNING_RATE=0.0003
EPOCHS=10
```

### Modifying Model Hyperparameters

Edit `config/settings.py` or set environment variables before training:

```bash
# Example: Train with larger model
export D_MODEL=64
export NUM_LAYERS=8
export EPOCHS=20
python train.py --train
```

## Data Generation Details

### Using the Data Augmentation Generator

The project includes `src/data/data_augmentation_generator.py` for SDV-based data generation (requires trained distribution model):

```python
from sdv.single_table import GaussianCopulaSynthesizer
import pandas as pd

# Load pre-trained synthesizer
synthesizer = GaussianCopulaSynthesizer.load('./distribution_model.pkl')

# Generate synthetic data
synthetic_data = synthesizer.sample(num_rows=10000)
synthetic_data.to_csv('./data/raw/synthetic_data.csv', index=False, encoding='utf-8-sig')
```

**Note:** The `generate_raw_data.py` script provides a simpler, self-contained approach that doesn't require a pre-trained SDV model.

### Custom Data Integration

To use your own data instead of synthetic data:

1. Prepare CSV file with same schema as `synthetic_data.csv`
2. Place in `data/raw/` directory
3. Update `--data-path` argument when training:

```bash
python train.py --train --data-path ./data/raw/your_data.csv
```

Or update data path in DAG files:
```python
# In src/airflow/dags/data_ingestion_dag.py
data_path = "./data/raw/your_data.csv"
```

## Testing

Run the test suite:

```bash
# All tests
pytest tests/

# Specific test file
pytest tests/test_kafka.py
pytest tests/test_models.py
pytest tests/test_mongodb.py

# With coverage report
pytest --cov=src tests/

# Verbose output
pytest -v tests/
```

## Troubleshooting

### Common Issues

**1. Data file not found**
```bash
FileNotFoundError: Data file not found: ./data/raw/synthetic_data.csv
```
**Solution:** Run `python generate_raw_data.py` to generate the data file.

**2. MongoDB connection issues**
```bash
# Check MongoDB is running
docker ps | grep mongodb

# Test connection
docker exec -it mongodb mongosh
```

**3. Airflow DAG not appearing**
```bash
# Check DAG syntax
python src/airflow/dags/data_ingestion_dag.py

# Refresh DAGs in UI
# Airflow UI -> DAGs -> Refresh button

# View scheduler logs
docker-compose logs airflow-scheduler
```

### Viewing Logs

```bash
# Application logs
tail -f logs/training_*.log
tail -f logs/data_generation_*.log

# Docker service logs
docker-compose logs -f airflow-scheduler
docker-compose logs -f airflow-webserver
docker-compose logs -f mongodb
```

### Cleaning Up

```bash
# Stop all services
docker-compose down

# Remove volumes (WARNING: deletes all data)
docker-compose down -v

# Clean generated data and models
rm -rf data/raw/*.csv models/*.pth logs/*.log
```

## Advanced Usage

### Accessing MongoDB Data

```bash
# Connect to MongoDB (if using Docker)
docker exec -it mongodb mongosh

# View collections
use mlops_default_prediction
show collections

# Query data
db.raw_data.find().limit(5)
db.model_metadata.find().pretty()
db.training_events.find().sort({timestamp: -1}).limit(10)
db.evaluation_events.find().sort({timestamp: -1}).limit(10)
```

## Deployment

### Production Deployment Checklist

1. **Update environment variables** for production
2. **Configure MongoDB replica set** for high availability
3. **Scale Kafka brokers** for higher throughput
4. **Use Kubernetes** for Airflow workers (optional)
5. **Setup monitoring** with Prometheus/Grafana
6. **Enable authentication** on all services
7. **Configure backups** for MongoDB and models
8. **Setup CI/CD pipeline** for automated deployment

### Scaling Considerations

- **Kafka**: Add brokers and increase partition count
- **MongoDB**: Use sharding for large datasets
- **Airflow**: Use CeleryExecutor with multiple workers
- **Model Serving**: Deploy behind API gateway (FastAPI/Flask)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Generate test data: `python generate_raw_data.py`
4. Make your changes
5. Add tests for new functionality
6. Run tests: `pytest tests/`
7. Commit your changes (`git commit -am 'Add new feature'`)
8. Push to the branch (`git push origin feature/your-feature`)
9. Submit a pull request

## License

This project is licensed under the MIT License.

## Contact

For questions or issues, please open a GitHub issue or contact the maintainers.

## Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Apache Kafka Documentation](https://kafka.apache.org/documentation/)
- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [MongoDB Documentation](https://docs.mongodb.com/)
- [Tab Transformer Paper](https://arxiv.org/abs/2012.06678)
