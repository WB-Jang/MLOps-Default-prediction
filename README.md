# MLOps Default Prediction Pipeline

A comprehensive MLOps pipeline for loan default prediction using PyTorch, Airflow, and MongoDB.

# 배포

```실행 시
cd /AI_ML_DL/Projects/MLOps-Default-prediction
docker-compose down
docker compose up -d
```

```포트 사용 시
sudo lsof -i :5432
sudo systemctl stop postgresql
```

```test 방법
# 태스크를 테스트 모드로 직접 실행
docker compose exec airflow-scheduler airflow tasks test model_evaluation_pipeline load_latest_model 2026-01-21
docker compose exec airflow-scheduler airflow tasks test model_evaluation_pipeline evaluate_model 2026-01-21
docker compose exec airflow-scheduler airflow tasks test model_evaluation_pipeline check_model_performance 2026-01-21
docker compose exec airflow-scheduler airflow tasks test model_evaluation_pipeline prepare_retraining_data 2026-01-21
```

```dag 실행 내역 삭제
docker compose exec airflow-scheduler airflow dags delete model_evaluation_pipeline -y
```

```
# 구문 에러 확인
python -m py_compile src/data/data_gen_loader_processor.py && echo "✅ No syntax errors" || echo "❌ Syntax error found"

# 컨테이너 재시작
docker compose restart airflow-scheduler airflow-webserver

```
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
   - **Data Ingestion DAG**: Automated data generation using distribution model and processing
   - **Model Training DAG**: End-to-end training pipeline
   - **Model Evaluation DAG**: Automated evaluation with **3-retry retraining** and alert mechanism
   - **Model Deployment DAG**: Automated deployment of approved models

5. **Advanced Features**
   - **Automatic Retraining**: Up to 3 retries if model F1 score < threshold
   - **Alert System**: Automatic alerts stored in MongoDB when max retries exceeded
   - **Code Reuse**: Evaluation and fine-tuning use code from `corp_default_modeling_f.py`
   - **Version Control**: All models tracked with versions in MongoDB
   - **Daily Automation**: Complete pipeline runs daily

## Project Structure

```
MLOps-Default-prediction/
├── src/
│   ├── models/           # Model architecture and training
│   │   ├── network.py    # Neural network definitions
│   │   ├── training.py   # Training utilities
│   │   └── evaluation.py # Evaluation & fine-tuning (from corp_default_modeling_f.py)
│   ├── data/             # Data processing
│   │   ├── data_gen_loader_processor.py  # SDV-based data generation
│   │   └── distribution_model.pkl         # Distribution model for synthetic data
│   ├── database/         # MongoDB integration
│   │   └── mongodb.py
│   └── airflow/
│       └── dags/         # Airflow DAGs
│           ├── data_ingestion_dag.py      # Data generation & processing
│           ├── model_training_dag.py       # Model training
│           ├── model_evaluation_dag.py     # Evaluation with retry
│           └── model_deployment_dag.py     # Model deployment
├── config/               # Configuration
│   └── settings.py
├── logs/                # Application logs
├── docker/              # Docker configurations
│   ├── Dockerfile.airflow
│   └── Dockerfile.app
├── docker-compose.yml   # Full stack deployment
├── requirements.txt
├── README.md
└── WORKFLOW.md         # Complete workflow documentation (Korean)
```

### model training results
```
--- Starting V2: Contrastive Learning Pre-training ---
[Pretrain V2] Epoch 1, Loss: 2.0119

--- Starting Fine-tuning ---
[Finetune] Epoch  1 | Loss: 97.8080 | Acc: 0.9795 | AUC: 0.9634 | F1: 0.8461
[Finetune] Epoch  2 | Loss: 26.8114 | Acc: 0.9884 | AUC: 0.9751 | F1: 0.9079
[Finetune] Epoch  3 | Loss: 2.0275 | Acc: 0.9924 | AUC: 0.9777 | F1: 0.9375
   -> Test Accuracy: 0.9952, Test ROC-AUC: 0.9771, Test F1-score: 0.9596
---pretrained_model_f.pth 저장 완료---
---finetuned_model_f.pth 저장 완료---
```
### Project Flow

본 프로젝트는 일별로 반복되는 완전한 MLOps 파이프라인을 구현합니다:

```
1. 초기 모델 저장 (MongoDB)
   ↓
2. 일별 데이터 생성 (Distribution model 활용)
   → MongoDB 저장
   ↓
3. 데이터 전처리
   → MongoDB processed_data 저장
   ↓
4. 최신 모델 평가 (corp_default_modeling_f.py 코드 사용)
   ↓
5. F1 Score 체크
   ├─ F1 >= 0.75 → 배포
   └─ F1 < 0.75 → 재학습 시작 (최대 3회)
       ├─ 1차 재학습 → 평가
       ├─ 2차 재학습 → 평가 (필요시)
       ├─ 3차 재학습 → 평가 (필요시)
       └─ 3회 후에도 실패 → Alert 발송
   ↓
6. 승인된 모델 MongoDB에 새 버전으로 저장
   ↓
7. 모델 배포 (프로덕션)
   ↓
8. 다음 날 2번부터 반복
```

### Enhanced DAG Flow (With Retry Mechanism)

```
                                            ┌─ (F1 >= 0.75) ─┐
                                            │                 │
load_latest_model → evaluate_model → check_model_performance │
                                            │                 │
                                            │ (F1 < 0.75)     │
                                            ▼                 │
                                     prepare_retraining_data  │
                                            ▼                 │
                                      finetune_model          │
                                            ▼                 │
                                   evaluate_finetuned_model   │
                                            │                 │
                    ┌─── (retry<3) ────────┤                 │
                    │                       │                 │
                    │              (retry>=3 & F1<0.75)       │
                    │                       │                 │
                    ↓                       ↓                 │
          check_model_performance      send_alert            │
               (Loop back)                  │                 │
                                            └────────┬────────┘
                                                     ▼
                                           send_evaluation_results
                                                     ↓
                                           deploy_model
```

**주요 특징**:
- **최대 3회 재학습**: F1 score가 threshold 미만이면 최대 3회까지 자동 재학습
- **자동 Alert**: 3회 재학습 후에도 실패하면 MongoDB에 alert 저장
- **코드 재사용**: `corp_default_modeling_f.py`의 평가/학습 코드 사용
- **버전 관리**: 모든 재학습 모델은 새 버전으로 MongoDB에 저장
- **자동 배포**: 승인된 모델은 자동으로 프로덕션에 배포
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

### Complete Workflow Documentation

For detailed information about the complete daily pipeline workflow, see **[WORKFLOW.md](WORKFLOW.md)** (Korean).

This document includes:
- Complete workflow diagram with retry mechanism
- Detailed implementation of each step
- How evaluation and fine-tuning code from `corp_default_modeling_f.py` is used
- MongoDB collection structure
- Retry and alert mechanism details
- Troubleshooting guide

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

# Check alerts (for failed retraining)
db.alerts.find().sort({timestamp: -1})

# Check processed data
db.processed_data.find().limit(5)

# Check deployment history
db.deployment_events.find().sort({timestamp: -1})
```

### Monitoring Model Performance

```javascript
// MongoDB shell queries for monitoring

// Get latest model evaluation
db.evaluation_events.findOne({}, {sort: {timestamp: -1}})

// Get models that needed retraining
db.model_metadata.find({finetuned_from: {$exists: true}})

// Get all alerts
db.alerts.find({alert_type: "MODEL_PERFORMANCE_FAILURE"})

// Check retry history
db.evaluation_events.find({
  "retraining_info.was_retrained": true
}).sort({timestamp: -1})
```

### Adjusting F1 Threshold and Retry Count

Edit `src/airflow/dags/model_evaluation_dag.py`:

```python
# Change F1 threshold (default: 0.75)
F1_THRESHOLD = 0.80  # More strict

# Change max retry attempts (default: 3)
MAX_RETRAIN_ATTEMPTS = 5  # More retries
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
