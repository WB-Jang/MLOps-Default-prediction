"""Data preprocessing utilities."""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from loguru import logger


class DataPreprocessor:
    """Preprocessor for loan default prediction data."""
    
    def __init__(self):
        """Initialize data preprocessor."""
        self.scaler = StandardScaler()
        self.cat_max_dict: Optional[Dict[int, int]] = None
        self.categorical_columns: Optional[list] = None
        self.numerical_columns: Optional[list] = None
    
    def fit(
        self,
        data: pd.DataFrame,
        categorical_columns: list,
        numerical_columns: list
    ) -> None:
        """
        Fit the preprocessor on training data.
        
        Args:
            data: Training dataframe
            categorical_columns: List of categorical column names
            numerical_columns: List of numerical column names
        """
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        
        # Fit scaler on numerical features
        if numerical_columns:
            self.scaler.fit(data[numerical_columns])
        
        # Build categorical max dictionary
        self.cat_max_dict = {}
        for idx, col in enumerate(categorical_columns):
            self.cat_max_dict[idx] = int(data[col].max()) + 1
        
        logger.info(
            f"Preprocessor fitted: {len(categorical_columns)} categorical, "
            f"{len(numerical_columns)} numerical features"
        )
    
    def transform(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform data using fitted preprocessor.
        
        Args:
            data: Dataframe to transform
            
        Returns:
            Tuple of (categorical_features, numerical_features)
        """
        if self.categorical_columns is None or self.numerical_columns is None:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")
        
        # Extract categorical features
        x_cat = data[self.categorical_columns].values.astype(np.int64)
        
        # Extract and scale numerical features
        x_num = data[self.numerical_columns].values.astype(np.float32)
        if len(self.numerical_columns) > 0:
            x_num = self.scaler.transform(x_num)
        
        return x_cat, x_num
    
    def fit_transform(
        self,
        data: pd.DataFrame,
        categorical_columns: list,
        numerical_columns: list
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit and transform data in one step.
        
        Args:
            data: Training dataframe
            categorical_columns: List of categorical column names
            numerical_columns: List of numerical column names
            
        Returns:
            Tuple of (categorical_features, numerical_features)
        """
        self.fit(data, categorical_columns, numerical_columns)
        return self.transform(data)
    
    def get_cat_max_dict(self) -> Dict[int, int]:
        """
        Get the categorical max dictionary.
        
        Returns:
            Dictionary mapping categorical feature index to max value + 1
        """
        if self.cat_max_dict is None:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")
        return self.cat_max_dict


def prepare_data_for_kafka(data: pd.DataFrame) -> Dict:
    """
    Convert dataframe to dictionary format suitable for Kafka.
    
    Args:
        data: Dataframe to convert
        
    Returns:
        Dictionary representation of the data
    """
    return data.to_dict(orient='records')


def parse_kafka_data(kafka_message: Dict) -> pd.DataFrame:
    """
    Parse Kafka message into dataframe.
    
    Args:
        kafka_message: Message received from Kafka
        
    Returns:
        Dataframe with parsed data
    """
    if isinstance(kafka_message, list):
        return pd.DataFrame(kafka_message)
    elif isinstance(kafka_message, dict):
        return pd.DataFrame([kafka_message])
    else:
        raise ValueError(f"Unsupported message format: {type(kafka_message)}")
