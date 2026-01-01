#!/bin/bash
# Setup script for MLOps Loan Default Prediction

set -e

echo "====================================="
echo "MLOps Loan Default Prediction Setup"
echo "====================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Error: Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file from .env.example..."
    cp .env.example .env
    echo "Please update .env file with your configuration."
fi

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p models data/raw data/processed airflow/logs tests

# Create .gitkeep files
touch data/raw/.gitkeep
touch data/processed/.gitkeep

# Build and start services
echo "Building Docker images..."
docker-compose build

echo "Starting services..."
docker-compose up -d

echo "Waiting for PostgreSQL to be ready..."
sleep 10

# Check service status
echo "Checking service status..."
docker-compose ps

echo ""
echo "====================================="
echo "Setup completed!"
echo "====================================="
echo ""
echo "Services:"
echo "  - PostgreSQL: localhost:5432"
echo "  - Airflow Webserver: http://localhost:8080"
echo "  - FastAPI Application: http://localhost:8000"
echo "  - API Documentation: http://localhost:8000/docs"
echo ""
echo "Next steps:"
echo "  1. Access Airflow UI at http://localhost:8080"
echo "  2. Enable the DAGs"
echo "  3. Trigger 'model_training' DAG to train the first model"
echo "  4. Check API at http://localhost:8000/docs"
echo ""
echo "To stop services: docker-compose down"
echo "To view logs: docker-compose logs -f [service_name]"
echo ""
