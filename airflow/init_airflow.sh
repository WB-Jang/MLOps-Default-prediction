#!/bin/bash
# Initialize Airflow

echo "Initializing Airflow..."

# Initialize the database
airflow db init

# Create admin user
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin

echo "Airflow initialized successfully!"
echo "Username: admin"
echo "Password: admin"
