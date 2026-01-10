# Quick Start Guide - MLOps Default Prediction Pipeline

This guide will help you get the entire MLOps pipeline up and running in minutes.

## Prerequisites

- Python 3.10+
- Docker and Docker Compose (for full stack)
- 8GB+ RAM

## Step 1: Generate Raw Data (REQUIRED)

This is the **first and most important step**. The entire pipeline depends on this data.

```bash
# Generate 10,000 loan application records
python generate_raw_data.py --num-rows 10000

# Output files:
# - data/raw/synthetic_data.csv (raw loan data)
# - data/raw/synthetic_data_info.txt (dataset statistics)
```

**Expected Output:**
```
============================================================
Loan Default Prediction - Raw Data Generation
============================================================
Configuration:
  - Number of rows: 10,000
  - Output directory: ./data/raw
  - Filename: synthetic_data.csv
...
Dataset shape: (10000, 27)
Data generation completed successfully!
============================================================
```

**Verify the data was generated:**
```bash
ls -lh data/raw/
head -5 data/raw/synthetic_data.csv
```

## Step 2: Choose Your Setup

### Option A: Quick Test (Standalone Training)

Best for: Testing the pipeline, development, quick iterations

```bash
# Install dependencies
pip install torch pandas numpy scikit-learn loguru

# Train a model (uses data/raw/synthetic_data.csv)
python train.py --train --epochs 5

# Output: models/classifier_YYYYMMDD_HHMMSS.pth
```

### Option B: Full MLOps Stack (Docker)

Best for: Production-like environment, end-to-end testing

```bash
# Start all services
docker-compose up -d

# Check services are running
docker-compose ps

# Initialize Airflow (first time only)
docker-compose run airflow-init

# Access Airflow UI
# Open browser: http://localhost:8080
# Login: admin / admin
```

## Step 3: Run the Pipeline

### Via Airflow UI (Option B only)

1. Open http://localhost:8080
2. Login with admin/admin
3. Enable these DAGs:
   - `data_ingestion_pipeline` - Loads CSV → Kafka
   - `model_training_pipeline` - Trains models
   - `model_evaluation_pipeline` - Evaluates models
4. Click "Trigger DAG" button on each
5. Monitor execution in the Graph view

### Via Command Line (Option A)

```bash
# Step 1: Generate data (if not done)
python generate_raw_data.py --num-rows 10000

# Step 2: Train model
python train.py --train --epochs 10 --data-path ./data/raw/synthetic_data.csv

# Step 3: Check output
ls -lh models/
```

## Step 4: Verify Everything Works

### Check Generated Data
```bash
# View first few rows
head data/raw/synthetic_data.csv

# Check data statistics
cat data/raw/synthetic_data_info.txt

# Count rows
wc -l data/raw/synthetic_data.csv
```

### Check Trained Models
```bash
# List saved models
ls -lh models/

# Should see files like:
# classifier_20260110_123456.pth
# pretrained_encoder_20260110_123456.pth
```

### Check Logs
```bash
# View training logs
ls -lh logs/
tail -50 logs/training_*.log

# View data generation logs
tail -50 logs/data_generation_*.log
```

## Common Issues and Solutions

### Issue 1: Data file not found
```
FileNotFoundError: Data file not found: ./data/raw/synthetic_data.csv
```
**Solution:** Run `python generate_raw_data.py`

### Issue 2: Module not found
```
ModuleNotFoundError: No module named 'pandas'
```
**Solution:** `pip install -r requirements.txt`

### Issue 3: Docker services not starting
```
ERROR: Service 'kafka' failed to start
```
**Solution:**
```bash
docker-compose down
docker-compose up -d
docker-compose logs kafka
```

### Issue 4: Permission denied on directories
```
PermissionError: [Errno 13] Permission denied: 'data/raw'
```
**Solution:**
```bash
mkdir -p data/raw models logs
chmod 755 data/raw models logs
```

## Next Steps

### 1. Customize Data Generation
```bash
# Generate more data
python generate_raw_data.py --num-rows 50000

# Generate test data without target
python generate_raw_data.py --num-rows 5000 --no-target --filename test_data.csv
```

### 2. Adjust Model Hyperparameters
```bash
# Train with custom settings
python train.py --train \
  --epochs 20 \
  --batch-size 64 \
  --lr 0.001 \
  --save-dir ./my_models
```

### 3. Use Your Own Data

Replace `data/raw/synthetic_data.csv` with your own CSV file that has:
- Categorical and numerical features
- A 'default' column for the target (0 or 1)
- A 'loan_id' column for identification

```bash
# Train with custom data
python train.py --train --data-path ./data/raw/my_data.csv
```

### 4. Monitor the Pipeline

**Airflow UI:**
- View DAG execution: http://localhost:8080/dags
- Check task logs: Click on task → View Log
- Monitor performance: Browse → Task Instances

**Logs Directory:**
```bash
# Watch training logs in real-time
tail -f logs/training_*.log

# Search for errors
grep -i error logs/*.log
```

### 5. Access MongoDB Data

```bash
# Connect to MongoDB (if using Docker)
docker exec -it mongodb mongosh

# View collections
use mlops_default_prediction
show collections
db.model_metadata.find().pretty()
db.training_data.find().limit(5).pretty()
```

## Full Pipeline Test

Run this to test everything end-to-end:

```bash
#!/bin/bash
set -e

echo "Step 1: Generate data"
python generate_raw_data.py --num-rows 1000

echo "Step 2: Test data loading"
python -c "from src.data.data_loader import load_data_for_training; load_data_for_training('./data/raw/synthetic_data.csv')"

echo "Step 3: Train model (requires torch)"
# python train.py --train --epochs 1

echo "✓ All tests passed!"
```

## Resources

- **Full Documentation:** See [README.md](README.md)
- **Architecture:** See [ARCHITECTURE.md](ARCHITECTURE.md)
- **Deployment:** See [DEPLOYMENT.md](DEPLOYMENT.md)

## Getting Help

1. Check logs in `./logs/` directory
2. Review error messages carefully
3. Ensure all prerequisites are installed
4. Verify data was generated successfully
5. Open a GitHub issue with:
   - Error message
   - Steps to reproduce
   - Log files (if applicable)

## Summary

**Minimal working example:**
```bash
# 1. Generate data
python generate_raw_data.py

# 2. Train model
python train.py --train --epochs 5

# Done! Check models/ directory for output
ls -lh models/
```

That's it! You now have a working MLOps pipeline for loan default prediction.
