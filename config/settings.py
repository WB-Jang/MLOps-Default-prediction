"""Configuration settings for the MLOps pipeline."""
from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    # MongoDB Configuration
    mongodb_host: str = Field(default="localhost", alias="MONGODB_HOST")
    mongodb_port: int = Field(default=27017, alias="MONGODB_PORT")
    mongodb_database: str = Field(default="mlops_default_prediction", alias="MONGODB_DATABASE")
    mongodb_username: Optional[str] = Field(default=None, alias="MONGODB_USERNAME")
    mongodb_password: Optional[str] = Field(default=None, alias="MONGODB_PASSWORD")
    
    # Kafka Configuration
    kafka_bootstrap_servers: str = Field(default="localhost:9092", alias="KAFKA_BOOTSTRAP_SERVERS")
    kafka_raw_data_topic: str = Field(default="raw_data", alias="KAFKA_RAW_DATA_TOPIC")
    kafka_processed_data_topic: str = Field(default="processed_data", alias="KAFKA_PROCESSED_DATA_TOPIC")
    kafka_commands_topic: str = Field(default="commands", alias="KAFKA_COMMANDS_TOPIC")
    kafka_consumer_group: str = Field(default="mlops_consumer_group", alias="KAFKA_CONSUMER_GROUP")
    
    # Model Configuration
    model_save_path: str = Field(default="./models", alias="MODEL_SAVE_PATH")
    model_format: str = Field(default="pth", alias="MODEL_FORMAT")
    d_model: int = Field(default=32, alias="D_MODEL")
    nhead: int = Field(default=4, alias="NHEAD")
    num_layers: int = Field(default=6, alias="NUM_LAYERS")
    dim_feedforward: int = Field(default=64, alias="DIM_FEEDFORWARD")
    dropout_rate: float = Field(default=0.3, alias="DROPOUT_RATE")
    
    # Training Configuration
    batch_size: int = Field(default=32, alias="BATCH_SIZE")
    learning_rate: float = Field(default=3e-4, alias="LEARNING_RATE")
    epochs: int = Field(default=10, alias="EPOCHS")
    pretrain_epochs: int = Field(default=10, alias="PRETRAIN_EPOCHS")
    temperature: float = Field(default=0.5, alias="TEMPERATURE")
    
    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    
    @property
    def mongodb_connection_string(self) -> str:
        """Build MongoDB connection string."""
        if self.mongodb_username and self.mongodb_password:
            # 데이터베이스 이름 제거 - 인증은 admin DB에서 수행
            return f"mongodb://{self.mongodb_username}:{self.mongodb_password}@{self.mongodb_host}:{self.mongodb_port}/"
        return f"mongodb://{self.mongodb_host}:{self.mongodb_port}/"


# Global settings instance
settings = Settings()
