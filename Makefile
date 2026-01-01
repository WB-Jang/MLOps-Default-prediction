.PHONY: help setup start stop logs clean test format lint

help:
	@echo "Loan Default Prediction MLOps - Available Commands"
	@echo ""
	@echo "  make setup    - Setup the environment and create necessary files"
	@echo "  make start    - Start all services"
	@echo "  make stop     - Stop all services"
	@echo "  make logs     - View logs from all services"
	@echo "  make clean    - Remove all containers and volumes"
	@echo "  make test     - Run tests"
	@echo "  make format   - Format code with black and isort"
	@echo "  make lint     - Run linting with flake8"

setup:
	@echo "Setting up MLOps environment..."
	@./setup.sh

start:
	@echo "Starting services..."
	docker-compose up -d
	@echo "Services started. Access:"
	@echo "  - Airflow: http://localhost:8080"
	@echo "  - FastAPI: http://localhost:8000"
	@echo "  - API Docs: http://localhost:8000/docs"

stop:
	@echo "Stopping services..."
	docker-compose down

restart:
	@echo "Restarting services..."
	docker-compose restart

logs:
	docker-compose logs -f

logs-app:
	docker-compose logs -f app

logs-airflow:
	docker-compose logs -f airflow-scheduler airflow-webserver

logs-postgres:
	docker-compose logs -f postgres

clean:
	@echo "Cleaning up..."
	docker-compose down -v
	@echo "Cleaned up all containers and volumes"

build:
	@echo "Building Docker images..."
	docker-compose build

test:
	@echo "Running tests..."
	poetry run pytest tests/ -v

format:
	@echo "Formatting code..."
	poetry run black src/ tests/
	poetry run isort src/ tests/

lint:
	@echo "Running linting..."
	poetry run flake8 src/ tests/

install:
	@echo "Installing dependencies..."
	poetry install

shell:
	docker-compose exec app /bin/bash

db-shell:
	docker exec -it loan_postgres psql -U mlops_user -d loan_default

airflow-shell:
	docker-compose exec airflow-webserver /bin/bash
