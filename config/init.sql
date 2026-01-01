-- Create airflow database for Airflow
CREATE DATABASE airflow_db;

-- Connect to loan_default database
\c loan_default;

-- Create schema for loan data
CREATE SCHEMA IF NOT EXISTS loan_data;

-- Table for raw loan data
CREATE TABLE IF NOT EXISTS loan_data.raw_data (
    id SERIAL PRIMARY KEY,
    data_date DATE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    categorical_features JSONB,
    numerical_features JSONB,
    target INTEGER,
    UNIQUE(data_date, categorical_features, numerical_features)
);

-- Table for processed/featured data
CREATE TABLE IF NOT EXISTS loan_data.processed_data (
    id SERIAL PRIMARY KEY,
    raw_data_id INTEGER REFERENCES loan_data.raw_data(id),
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    features JSONB,
    target INTEGER
);

-- Table for model metadata
CREATE TABLE IF NOT EXISTS loan_data.model_metadata (
    id SERIAL PRIMARY KEY,
    model_version VARCHAR(50) UNIQUE NOT NULL,
    model_path VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    training_samples INTEGER,
    f1_score FLOAT,
    accuracy FLOAT,
    precision_score FLOAT,
    recall FLOAT,
    roc_auc FLOAT,
    is_active BOOLEAN DEFAULT false,
    training_config JSONB
);

-- Table for model predictions
CREATE TABLE IF NOT EXISTS loan_data.predictions (
    id SERIAL PRIMARY KEY,
    model_version VARCHAR(50) REFERENCES loan_data.model_metadata(model_version),
    data_id INTEGER REFERENCES loan_data.raw_data(id),
    predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    prediction INTEGER,
    probability FLOAT,
    actual_outcome INTEGER
);

-- Table for model performance monitoring
CREATE TABLE IF NOT EXISTS loan_data.model_performance (
    id SERIAL PRIMARY KEY,
    model_version VARCHAR(50) REFERENCES loan_data.model_metadata(model_version),
    evaluation_date DATE NOT NULL,
    f1_score FLOAT,
    accuracy FLOAT,
    precision_score FLOAT,
    recall FLOAT,
    roc_auc FLOAT,
    sample_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_raw_data_date ON loan_data.raw_data(data_date);
CREATE INDEX idx_predictions_model ON loan_data.predictions(model_version);
CREATE INDEX idx_predictions_date ON loan_data.predictions(predicted_at);
CREATE INDEX idx_performance_date ON loan_data.model_performance(evaluation_date);
CREATE INDEX idx_model_active ON loan_data.model_metadata(is_active);

-- Grant permissions
GRANT ALL PRIVILEGES ON SCHEMA loan_data TO mlops_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA loan_data TO mlops_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA loan_data TO mlops_user;
