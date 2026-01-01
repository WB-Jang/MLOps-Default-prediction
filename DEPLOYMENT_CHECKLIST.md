# Deployment Checklist

Use this checklist when deploying the MLOps system.

## Pre-Deployment

### Infrastructure
- [ ] Docker and Docker Compose installed
- [ ] Sufficient system resources (4GB+ RAM, 10GB+ disk)
- [ ] Required ports available (5432, 8000, 8080)
- [ ] Network connectivity verified

### Configuration
- [ ] `.env` file created from `.env.example`
- [ ] Database credentials updated
- [ ] Airflow secret key changed
- [ ] Model path configured
- [ ] Threshold values reviewed (F1_SCORE_THRESHOLD=0.8)

### Security
- [ ] Changed default PostgreSQL password
- [ ] Changed default Airflow admin password
- [ ] Reviewed network exposure settings
- [ ] Configured firewall rules (if production)

## Initial Setup

### Services
- [ ] Run `./setup.sh` successfully
- [ ] All containers started (`docker-compose ps`)
- [ ] PostgreSQL healthy and accepting connections
- [ ] Airflow webserver accessible (http://localhost:8080)
- [ ] API accessible (http://localhost:8000)

### Database
- [ ] Database schema initialized
- [ ] Tables created successfully
- [ ] Indexes created
- [ ] Permissions granted

### Airflow
- [ ] Airflow initialized (`airflow db init`)
- [ ] Admin user created
- [ ] DAGs visible in UI
- [ ] No DAG errors

## Data Setup

### Sample Data (for testing)
- [ ] Generated sample data: `python scripts/generate_sample_data.py 1000`
- [ ] Data visible in database
- [ ] Data validation passed

### Production Data (for production)
- [ ] Data source configured in `data_collection_dag.py`
- [ ] Data collection DAG tested
- [ ] Data validation logic implemented
- [ ] Data quality checks in place

## Model Training

### First Training
- [ ] Training DAG triggered
- [ ] Pretraining completed successfully
- [ ] Classifier training completed
- [ ] Model saved to `models/` directory
- [ ] Model metadata saved to database
- [ ] Training metrics look reasonable

### Model Validation
- [ ] Model loaded successfully
- [ ] Validation metrics calculated
- [ ] F1-score meets threshold (>= 0.8)
- [ ] Model set as active

## API Testing

### Health Checks
- [ ] `GET /health` returns 200
- [ ] Model version displayed correctly
- [ ] Model loaded status is true

### Predictions
- [ ] `POST /predict` accepts requests
- [ ] Predictions return valid probabilities (0-1)
- [ ] Predictions logged to database
- [ ] Response time acceptable (< 100ms)

### Model Management
- [ ] `GET /model-info` returns correct data
- [ ] `POST /reload-model` works correctly

## Automation

### Data Collection
- [ ] DAG scheduled correctly (@daily)
- [ ] Data collection runs successfully
- [ ] Data validation works
- [ ] Failed runs trigger alerts (if configured)

### Model Training
- [ ] DAG can be triggered manually
- [ ] Training runs end-to-end
- [ ] Models versioned correctly
- [ ] Training failures handled gracefully

### Validation & Deployment
- [ ] DAG scheduled correctly (@daily)
- [ ] Model evaluation works
- [ ] F1-score threshold check working
- [ ] Retraining triggered when needed
- [ ] New models deployed automatically

## Monitoring

### Logs
- [ ] Application logs accessible
- [ ] Airflow logs accessible
- [ ] PostgreSQL logs accessible
- [ ] Log rotation configured (if needed)

### Metrics
- [ ] Model performance tracked in database
- [ ] System metrics monitored (if tools configured)
- [ ] Alert thresholds configured (if needed)

### Database
- [ ] Database backups configured
- [ ] Backup restoration tested
- [ ] Database size monitored
- [ ] Query performance acceptable

## Documentation

### Internal
- [ ] Team trained on system usage
- [ ] Runbook created for operations
- [ ] Contact information documented
- [ ] Escalation procedures defined

### External
- [ ] API documentation accessible
- [ ] README reviewed and updated
- [ ] Architecture diagram updated (if needed)
- [ ] Change log maintained

## Production Readiness

### Performance
- [ ] Load testing completed (if production)
- [ ] Response times meet SLA
- [ ] Resource usage acceptable
- [ ] Scaling strategy defined

### Security
- [ ] SSL/TLS enabled (if production)
- [ ] Authentication implemented (if needed)
- [ ] Authorization configured (if needed)
- [ ] Security scan completed (if tools available)
- [ ] Secrets management configured

### Reliability
- [ ] Backup strategy implemented
- [ ] Disaster recovery plan documented
- [ ] High availability considered (if needed)
- [ ] Rollback procedure tested

### Compliance
- [ ] Data privacy requirements met
- [ ] Audit logging configured (if needed)
- [ ] Retention policies implemented
- [ ] Regulatory requirements satisfied

## Post-Deployment

### Verification
- [ ] End-to-end test successful
- [ ] Smoke tests passed
- [ ] Integration tests passed
- [ ] User acceptance testing completed

### Monitoring
- [ ] First 24 hours monitored closely
- [ ] No critical errors in logs
- [ ] Performance metrics normal
- [ ] Users able to access system

### Communication
- [ ] Stakeholders notified
- [ ] Documentation shared
- [ ] Support team briefed
- [ ] Users trained (if needed)

## Maintenance

### Daily
- [ ] Check DAG execution status
- [ ] Review error logs
- [ ] Monitor system resources
- [ ] Verify backup completion

### Weekly
- [ ] Review model performance trends
- [ ] Check for data quality issues
- [ ] Review API usage patterns
- [ ] Update documentation as needed

### Monthly
- [ ] Database maintenance (vacuum, analyze)
- [ ] Review and update dependencies
- [ ] Test backup restoration
- [ ] Review and optimize resources

### Quarterly
- [ ] Security review and updates
- [ ] Performance optimization review
- [ ] Architecture review
- [ ] Disaster recovery drill

## Troubleshooting

### Common Issues
- [ ] Port conflicts - Resolution documented
- [ ] Database connection issues - Resolution documented
- [ ] Memory issues - Resolution documented
- [ ] Model loading failures - Resolution documented

### Support Resources
- [ ] Support contact information documented
- [ ] Escalation procedures defined
- [ ] Knowledge base created
- [ ] FAQ documented

## Sign-Off

Deployment completed by: _________________ Date: _________

Verified by: _________________ Date: _________

Notes:
_________________________________________________________________
_________________________________________________________________
_________________________________________________________________
