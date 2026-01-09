# Deployment Guide

This guide covers deploying the MLOps Default Prediction pipeline in various environments.

## Table of Contents
- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Production Deployment](#production-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Troubleshooting](#troubleshooting)

## Local Development

### Prerequisites
- Python 3.10+
- pip or conda
- Git

### Steps

1. **Clone repository**
   ```bash
   git clone https://github.com/WB-Jang/MLOps-Default-prediction.git
   cd MLOps-Default-prediction
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup environment variables**
   ```bash
   cp .env.example .env
   # Edit .env file with your configuration
   ```

5. **Start individual services**

   **Kafka (using Docker)**
   ```bash
   docker run -d --name zookeeper -p 2181:2181 confluentinc/cp-zookeeper:7.5.0
   docker run -d --name kafka -p 9092:9092 \
     -e KAFKA_ZOOKEEPER_CONNECT=localhost:2181 \
     -e KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://localhost:9092 \
     confluentinc/cp-kafka:7.5.0
   ```

   **MongoDB (using Docker)**
   ```bash
   docker run -d --name mongodb -p 27017:27017 \
     -e MONGO_INITDB_ROOT_USERNAME=admin \
     -e MONGO_INITDB_ROOT_PASSWORD=changeme \
     mongo:7.0
   ```

   **Airflow (local)**
   ```bash
   export AIRFLOW_HOME=~/airflow
   airflow db init
   airflow users create --username admin --password admin \
     --firstname Admin --lastname User --role Admin \
     --email admin@example.com
   
   # Start webserver and scheduler in separate terminals
   airflow webserver --port 8080
   airflow scheduler
   ```

6. **Run training script**
   ```bash
   python train.py --pretrain --train
   ```

## Docker Deployment

### Full Stack with Docker Compose

This is the recommended approach for development and testing.

1. **Prerequisites**
   - Docker 20.10+
   - Docker Compose 2.0+
   - 8GB+ RAM

2. **Clone and configure**
   ```bash
   git clone https://github.com/WB-Jang/MLOps-Default-prediction.git
   cd MLOps-Default-prediction
   cp .env.example .env
   ```

3. **Build and start services**
   ```bash
   docker-compose build
   docker-compose up -d
   ```

4. **Initialize Airflow (first time only)**
   ```bash
   docker-compose run airflow-init
   ```

5. **Verify services**
   ```bash
   # Check all containers are running
   docker-compose ps
   
   # Expected output:
   # - zookeeper: Up
   # - kafka: Up
   # - mongodb: Up
   # - postgres: Up
   # - airflow-webserver: Up
   # - airflow-scheduler: Up
   ```

6. **Access services**
   - Airflow UI: http://localhost:8080 (admin/admin)
   - MongoDB: mongodb://admin:changeme@localhost:27017
   - Kafka: localhost:9092

7. **View logs**
   ```bash
   # All services
   docker-compose logs -f
   
   # Specific service
   docker-compose logs -f airflow-scheduler
   ```

8. **Stop services**
   ```bash
   docker-compose down
   
   # Remove volumes (caution: deletes data)
   docker-compose down -v
   ```

## Production Deployment

### Configuration Updates

1. **Update .env for production**
   ```bash
   # Use strong passwords
   MONGODB_USERNAME=prod_user
   MONGODB_PASSWORD=<strong-password>
   
   # Use production Kafka cluster
   KAFKA_BOOTSTRAP_SERVERS=kafka1:9092,kafka2:9092,kafka3:9092
   
   # Enable authentication
   AIRFLOW__API__AUTH_BACKENDS=airflow.api.auth.backend.basic_auth
   ```

2. **Update docker-compose.yml**
   ```yaml
   # Use production images
   services:
     airflow-webserver:
       image: your-registry/mlops-airflow:v1.0.0
       
     # Add resource limits
     deploy:
       resources:
         limits:
           cpus: '2'
           memory: 4G
         reservations:
           cpus: '1'
           memory: 2G
   ```

### Kafka Production Setup

1. **Multi-broker cluster**
   ```yaml
   # Add to docker-compose.yml or use managed Kafka
   kafka-1:
     image: confluentinc/cp-kafka:7.5.0
     environment:
       KAFKA_BROKER_ID: 1
       # ... configuration
   
   kafka-2:
     image: confluentinc/cp-kafka:7.5.0
     environment:
       KAFKA_BROKER_ID: 2
       # ... configuration
   
   kafka-3:
     image: confluentinc/cp-kafka:7.5.0
     environment:
       KAFKA_BROKER_ID: 3
       # ... configuration
   ```

2. **Create topics with replication**
   ```bash
   docker exec kafka kafka-topics --create \
     --topic raw_data \
     --bootstrap-server localhost:9092 \
     --partitions 3 \
     --replication-factor 3
   
   docker exec kafka kafka-topics --create \
     --topic processed_data \
     --bootstrap-server localhost:9092 \
     --partitions 3 \
     --replication-factor 3
   
   docker exec kafka kafka-topics --create \
     --topic commands \
     --bootstrap-server localhost:9092 \
     --partitions 3 \
     --replication-factor 3
   ```

### MongoDB Production Setup

1. **Enable authentication**
   ```bash
   # In docker-compose.yml or managed MongoDB
   mongodb:
     image: mongo:7.0
     environment:
       MONGO_INITDB_ROOT_USERNAME: admin
       MONGO_INITDB_ROOT_PASSWORD: ${MONGODB_PASSWORD}
     command: --auth
   ```

2. **Setup replica set** (for high availability)
   ```yaml
   mongodb-1:
     image: mongo:7.0
     command: mongod --replSet rs0 --bind_ip_all
   
   mongodb-2:
     image: mongo:7.0
     command: mongod --replSet rs0 --bind_ip_all
   
   mongodb-3:
     image: mongo:7.0
     command: mongod --replSet rs0 --bind_ip_all
   ```

3. **Initialize replica set**
   ```javascript
   rs.initiate({
     _id: "rs0",
     members: [
       { _id: 0, host: "mongodb-1:27017" },
       { _id: 1, host: "mongodb-2:27017" },
       { _id: 2, host: "mongodb-3:27017" }
     ]
   })
   ```

### Airflow Production Setup

1. **Use CeleryExecutor for scaling**
   ```yaml
   # docker-compose.yml
   airflow-worker:
     build:
       context: .
       dockerfile: docker/Dockerfile.airflow
     command: celery worker
     environment:
       AIRFLOW__CORE__EXECUTOR: CeleryExecutor
       AIRFLOW__CELERY__BROKER_URL: redis://redis:6379/0
     deploy:
       replicas: 3  # Multiple workers
   
   redis:
     image: redis:7-alpine
     ports:
       - "6379:6379"
   ```

2. **Configure connections**
   ```bash
   # Airflow UI > Admin > Connections
   # Add MongoDB connection
   # Add Kafka connection (if using Kafka provider)
   ```

## Kubernetes Deployment

For large-scale production deployments, use Kubernetes.

### Prerequisites
- Kubernetes cluster (1.24+)
- kubectl configured
- Helm 3+

### Deploy Kafka

```bash
# Using Strimzi Kafka Operator
kubectl create namespace kafka
kubectl apply -f 'https://strimzi.io/install/latest?namespace=kafka' -n kafka

# Create Kafka cluster
cat <<EOF | kubectl apply -f -
apiVersion: kafka.strimzi.io/v1beta2
kind: Kafka
metadata:
  name: mlops-cluster
  namespace: kafka
spec:
  kafka:
    version: 3.5.0
    replicas: 3
    listeners:
      - name: plain
        port: 9092
        type: internal
        tls: false
    storage:
      type: persistent-claim
      size: 100Gi
  zookeeper:
    replicas: 3
    storage:
      type: persistent-claim
      size: 10Gi
EOF
```

### Deploy MongoDB

```bash
# Using MongoDB Operator
helm repo add mongodb https://mongodb.github.io/helm-charts
helm install mongodb mongodb/community-operator --namespace mongodb --create-namespace

# Create MongoDB cluster
cat <<EOF | kubectl apply -f -
apiVersion: mongodbcommunity.mongodb.com/v1
kind: MongoDBCommunity
metadata:
  name: mlops-mongodb
  namespace: mongodb
spec:
  members: 3
  type: ReplicaSet
  version: "7.0.0"
  security:
    authentication:
      modes: ["SCRAM"]
  users:
    - name: admin
      db: admin
      passwordSecretRef:
        name: mongodb-admin-password
      roles:
        - name: root
          db: admin
  statefulSet:
    spec:
      volumeClaimTemplates:
        - metadata:
            name: data-volume
          spec:
            accessModes: [ "ReadWriteOnce" ]
            resources:
              requests:
                storage: 100Gi
EOF
```

### Deploy Airflow

```bash
# Using official Airflow Helm chart
helm repo add apache-airflow https://airflow.apache.org
helm repo update

# Create values file
cat > airflow-values.yaml <<EOF
executor: "CeleryExecutor"
workers:
  replicas: 3
env:
  - name: MONGODB_HOST
    value: "mlops-mongodb.mongodb.svc.cluster.local"
  - name: KAFKA_BOOTSTRAP_SERVERS
    value: "mlops-cluster-kafka-bootstrap.kafka.svc.cluster.local:9092"
dags:
  gitSync:
    enabled: true
    repo: https://github.com/WB-Jang/MLOps-Default-prediction.git
    branch: main
    subPath: "src/airflow/dags"
EOF

# Install Airflow
helm install airflow apache-airflow/airflow \
  --namespace airflow \
  --create-namespace \
  -f airflow-values.yaml
```

## Troubleshooting

### Kafka Issues

**Problem**: Kafka broker not starting
```bash
# Check logs
docker logs kafka

# Common fix: Clear data and restart
docker-compose down -v
docker-compose up -d
```

**Problem**: Cannot connect to Kafka from container
```bash
# Check network
docker network inspect mlops-network

# Verify KAFKA_ADVERTISED_LISTENERS matches your setup
```

### MongoDB Issues

**Problem**: Authentication failed
```bash
# Verify credentials
docker exec -it mongodb mongo -u admin -p changeme --authenticationDatabase admin

# Reset if needed
docker-compose down
docker volume rm mlops_mongodb_data
docker-compose up -d
```

### Airflow Issues

**Problem**: DAGs not appearing
```bash
# Check DAG folder is mounted
docker exec -it airflow-scheduler ls /opt/airflow/dags

# Verify no Python errors
docker exec -it airflow-scheduler python /opt/airflow/dags/data_ingestion_dag.py
```

**Problem**: Tasks failing
```bash
# Check task logs in Airflow UI
# Check environment variables
docker exec -it airflow-scheduler env | grep KAFKA
docker exec -it airflow-scheduler env | grep MONGODB
```

### General Docker Issues

**Problem**: Out of memory
```bash
# Increase Docker memory limit (Docker Desktop)
# Settings > Resources > Memory > 8GB or more

# Or use resource limits
docker-compose up -d --scale airflow-worker=1
```

**Problem**: Port already in use
```bash
# Find process using port
lsof -i :8080

# Kill process or change port in docker-compose.yml
```

## Health Checks

### Verify Kafka
```bash
# List topics
docker exec kafka kafka-topics --list --bootstrap-server localhost:9092

# Test producer/consumer
docker exec kafka kafka-console-producer --topic test --bootstrap-server localhost:9092
docker exec kafka kafka-console-consumer --topic test --from-beginning --bootstrap-server localhost:9092
```

### Verify MongoDB
```bash
# Connect to MongoDB
docker exec -it mongodb mongo -u admin -p changeme

# Check databases
show dbs
use mlops_default_prediction
show collections
```

### Verify Airflow
```bash
# Check DAGs
curl http://localhost:8080/api/v1/dags -u admin:admin

# Trigger DAG
curl -X POST http://localhost:8080/api/v1/dags/data_ingestion_pipeline/dagRuns \
  -u admin:admin \
  -H "Content-Type: application/json" \
  -d '{}'
```

## Backup and Recovery

### MongoDB Backup
```bash
# Backup
docker exec mongodb mongodump --out /backup --username admin --password changeme

# Restore
docker exec mongodb mongorestore /backup --username admin --password changeme
```

### Model Files Backup
```bash
# Backup models directory
tar -czf models-backup-$(date +%Y%m%d).tar.gz models/

# Restore
tar -xzf models-backup-YYYYMMDD.tar.gz
```

## Performance Tuning

### Kafka Optimization
- Increase partitions for parallel processing
- Adjust `batch.size` and `linger.ms` for throughput
- Use compression (snappy or lz4)

### MongoDB Optimization
- Create indexes on frequently queried fields
- Use projection to fetch only needed fields
- Enable sharding for large collections

### Airflow Optimization
- Use connection pooling
- Adjust worker concurrency
- Optimize task parallelism

## Security Checklist

- [ ] Change default passwords
- [ ] Enable TLS/SSL for all connections
- [ ] Use secrets management (HashiCorp Vault, AWS Secrets Manager)
- [ ] Implement network policies in Kubernetes
- [ ] Enable audit logging
- [ ] Regular security updates
- [ ] Backup encryption
- [ ] Access control lists (ACLs) on Kafka topics
- [ ] MongoDB role-based access control (RBAC)
