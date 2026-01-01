# Quick Start Guide

Get up and running with the Loan Default Prediction MLOps system in 5 minutes!

## Prerequisites

- Docker and Docker Compose installed
- At least 4GB RAM available
- Ports 5432, 8000, and 8080 available

## Step 1: Clone and Setup

```bash
# Clone the repository
git clone https://github.com/WB-Jang/MLOps-Default-prediction.git
cd MLOps-Default-prediction

# Run setup script
./setup.sh
```

This will:
- Create necessary directories
- Copy environment configuration
- Build Docker images
- Start all services

## Step 2: Verify Services

Check that all services are running:

```bash
docker-compose ps
```

You should see:
- ✅ postgres (healthy)
- ✅ airflow-webserver (healthy)
- ✅ airflow-scheduler (running)
- ✅ app (running)

## Step 3: Access the Interfaces

### Airflow UI
- URL: http://localhost:8080
- Username: `admin`
- Password: `admin`

### API Documentation
- URL: http://localhost:8000/docs
- Interactive Swagger UI for testing endpoints

### API Health Check
```bash
curl http://localhost:8000/health
```

## Step 4: Initialize Airflow (First Time Only)

If this is your first time, initialize Airflow:

```bash
docker-compose exec airflow-webserver bash
./init_airflow.sh
exit
```

## Step 5: Run Your First Pipeline

### Option A: Using Airflow UI

1. Go to http://localhost:8080
2. Login with admin/admin
3. Enable the `model_training` DAG
4. Click the "Play" button to trigger it

### Option B: Using CLI

```bash
# Trigger training DAG
docker exec loan_airflow_scheduler airflow dags trigger model_training

# Check status
docker exec loan_airflow_scheduler airflow dags list-runs -d model_training
```

## Step 6: Make a Prediction

Once a model is trained and deployed:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "categorical_features": {
      "feature1": 1,
      "feature2": 2
    },
    "numerical_features": {
      "feature3": 0.5,
      "feature4": 1.2
    }
  }'
```

## Common Commands

### View Logs

```bash
# All services
make logs

# Specific service
make logs-app
make logs-airflow
make logs-postgres
```

### Stop Services

```bash
make stop
```

### Restart Services

```bash
make restart
```

### Clean Up

```bash
# Remove containers and volumes
make clean
```

## Troubleshooting

### Services Won't Start

```bash
# Check Docker resources
docker system df

# Restart Docker daemon
sudo systemctl restart docker

# Rebuild containers
make clean
make build
make start
```

### Can't Connect to Database

```bash
# Check PostgreSQL logs
make logs-postgres

# Verify database is ready
docker exec loan_postgres pg_isready -U mlops_user
```

### Airflow DAGs Not Showing

```bash
# Restart scheduler
docker-compose restart airflow-scheduler

# Check DAG folder permissions
ls -la airflow/dags/
```

### Port Already in Use

If you see "port already allocated" errors:

```bash
# Find process using port (example: 8080)
lsof -i :8080

# Kill the process or change port in docker-compose.yml
```

## Next Steps

1. **Customize Configuration**: Edit `.env` file for your needs
2. **Add Real Data**: Implement data collection in `airflow/dags/data_collection_dag.py`
3. **Configure Training**: Adjust hyperparameters in `config/settings.py`
4. **Set Up Monitoring**: Enable the validation DAG for automatic monitoring
5. **Read Documentation**: Check `README.md` and `ARCHITECTURE.md` for details

## Quick Reference

| Component | URL/Port | Credentials |
|-----------|----------|-------------|
| Airflow UI | http://localhost:8080 | admin/admin |
| FastAPI | http://localhost:8000 | - |
| API Docs | http://localhost:8000/docs | - |
| PostgreSQL | localhost:5432 | mlops_user/mlops_password |
| Database | loan_default | - |

## Getting Help

- Check logs: `make logs`
- View README: `cat README.md`
- View architecture: `cat ARCHITECTURE.md`
- Open an issue on GitHub

## Development Mode

For local development without Docker:

```bash
# Install dependencies
poetry install

# Set environment variables
export DATABASE_URL="postgresql://mlops_user:mlops_password@localhost:5432/loan_default"

# Run API locally
poetry run uvicorn src.api.main:app --reload

# Run tests
poetry run pytest tests/
```

---

**Success!** 🎉 You now have a fully functional MLOps pipeline running!
