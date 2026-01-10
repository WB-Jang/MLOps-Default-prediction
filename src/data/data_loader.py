"""Data loading utilities for the MLOps pipeline."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict
from loguru import logger


class DataLoader:
    """Load and manage raw data for the pipeline."""
    
    def __init__(self, data_path: str = "./data/raw/synthetic_data.csv"):
        """
        Initialize data loader.
        
        Args:
            data_path: Path to the raw CSV data file
        """
        self.data_path = Path(data_path)
        self.df: Optional[pd.DataFrame] = None
        self.categorical_columns: List[str] = []
        self.numerical_columns: List[str] = []
        
    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw data from CSV file.
        
        Returns:
            DataFrame with raw data
        """
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {self.data_path}\n"
                "Please run: python generate_raw_data.py"
            )
        
        logger.info(f"Loading raw data from {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(self.df)} rows with {len(self.df.columns)} columns")
        
        # Auto-detect categorical and numerical columns
        self._detect_column_types()
        
        return self.df
    
    def _detect_column_types(self) -> None:
        """Detect categorical and numerical columns automatically."""
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load_raw_data() first.")
        
        # Exclude ID and target columns
        exclude_cols = ['loan_id', 'default']
        
        for col in self.df.columns:
            if col in exclude_cols:
                continue
            
            # Check if column is categorical (object type or few unique values)
            if self.df[col].dtype == 'object' or self.df[col].nunique() < 20:
                self.categorical_columns.append(col)
            else:
                self.numerical_columns.append(col)
        
        logger.info(f"Detected {len(self.categorical_columns)} categorical columns")
        logger.info(f"Detected {len(self.numerical_columns)} numerical columns")
    
    def encode_categorical_columns(self) -> Dict[str, Dict]:
        """
        Encode categorical columns as integers.
        
        Returns:
            Dictionary mapping column names to encoding dictionaries
        """
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load_raw_data() first.")
        
        encoding_maps = {}
        
        for col in self.categorical_columns:
            if self.df[col].dtype == 'object':
                # Create encoding map
                unique_values = sorted(self.df[col].unique())
                encoding_map = {val: idx for idx, val in enumerate(unique_values)}
                encoding_maps[col] = encoding_map
                
                # Apply encoding
                self.df[col] = self.df[col].map(encoding_map)
                logger.debug(f"Encoded {col}: {len(encoding_map)} unique values")
        
        return encoding_maps
    
    def get_features_and_target(
        self,
        encode_categoricals: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Get features and target variable.
        
        Args:
            encode_categoricals: Whether to encode categorical variables
            
        Returns:
            Tuple of (features_df, target_series)
        """
        if self.df is None:
            self.load_raw_data()
        
        # Encode if needed
        if encode_categoricals:
            self.encode_categorical_columns()
        
        # Separate features and target
        if 'default' in self.df.columns:
            target = self.df['default']
            features = self.df.drop(columns=['loan_id', 'default'], errors='ignore')
        else:
            target = None
            features = self.df.drop(columns=['loan_id'], errors='ignore')
        
        return features, target
    
    def get_categorical_max_dict(self) -> Dict[int, int]:
        """
        Get the maximum value for each categorical feature (for embedding size).
        
        Returns:
            Dictionary mapping feature index to max value + 1
        """
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load_raw_data() first.")
        
        cat_max_dict = {}
        for idx, col in enumerate(self.categorical_columns):
            cat_max_dict[idx] = int(self.df[col].max()) + 1
        
        return cat_max_dict
    
    def train_test_split(
        self,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets.
        
        Args:
            test_size: Proportion of data for test set
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        from sklearn.model_selection import train_test_split
        
        X, y = self.get_features_and_target(encode_categoricals=True)
        
        if y is None:
            raise ValueError("Cannot split: target variable 'default' not found")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    def get_column_lists(self) -> Tuple[List[str], List[str]]:
        """
        Get lists of categorical and numerical column names.
        
        Returns:
            Tuple of (categorical_columns, numerical_columns)
        """
        if not self.categorical_columns or not self.numerical_columns:
            if self.df is None:
                self.load_raw_data()
            else:
                self._detect_column_types()
        
        return self.categorical_columns, self.numerical_columns
    
    def save_processed_data(self, output_path: str) -> None:
        """
        Save processed data to CSV.
        
        Args:
            output_path: Path to save the processed data
        """
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load_raw_data() first.")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.df.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")


def load_data_for_training(
    data_path: str = "./data/raw/synthetic_data.csv",
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series], Dict]:
    """
    Load and prepare data for model training.
    
    Args:
        data_path: Path to raw data CSV file
        test_size: Proportion for test set
        random_state: Random seed
        
    Returns:
        Tuple of ((X_train, y_train), (X_test, y_test), metadata)
    """
    loader = DataLoader(data_path)
    loader.load_raw_data()
    
    # Get train/test split
    X_train, X_test, y_train, y_test = loader.train_test_split(
        test_size=test_size,
        random_state=random_state
    )
    
    # Get metadata
    cat_cols, num_cols = loader.get_column_lists()
    cat_max_dict = loader.get_categorical_max_dict()
    
    metadata = {
        'categorical_columns': cat_cols,
        'numerical_columns': num_cols,
        'cat_max_dict': cat_max_dict,
        'num_categorical_features': len(cat_cols),
        'num_numerical_features': len(num_cols),
    }
    
    return (X_train, y_train), (X_test, y_test), metadata
