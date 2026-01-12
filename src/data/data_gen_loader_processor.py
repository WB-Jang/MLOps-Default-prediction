"""Data loading utilities for the MLOps pipeline."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict
from loguru import logger
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

class DataGenLoaderProcessor:
    """Generate and Load and manage raw data for the pipeline."""
    
    def __init__(self, data_path: str = "synthetic_data.csv"):
        """
        Initialize data loader.
        
        Args:
            data_path: Path to the raw CSV data file
        """
        self.data_path = Path(data_path)
        self.df: Optional[pd.DataFrame] = None
        self.categorical_columns: List[str] = []
        self.numerical_columns: List[str] = []
        
    def data_generation(self, data_quantity: int = 10000):
        synthesizer = GaussianCopulaSynthesizer.load('distribution_model.pkl')
        synthetic_data = synthesizer.sample(data_quantity)
        synthetic_data.to_csv(self.data_path,index=False, encoding='utf-8-sig')
        
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
                      
        return self.df

    def IsNull_cols(self):
        if not self.df:
            raise FileNotFoundError(
                f"df not found"
            )
        null_columns = self.df.columns[self.df.isnull().any()]
        print(f'null 값이 있는 컬럼 리스트 : {null_columns}')
        for col in null_columns:
            new_column = f'{col}_isnull'
            self.df[new_column] = self.df[col].isnull().map(lambda x:'Null' if x else 'Notnull')
        print('---IsNull 컬럼 생성 완료---')
        return self.df

    def obj_cols(self):
        if not self.df:
            raise FileNotFoundError(
                f"df not found"
            )
        object_dtype_list = self.df.select_dtypes(include='object').columns.tolist()
        object_dtype_list.append('acct_titl_cd')
        print(f'object 형식인 컬럼 리스트 : {object_dtype_list}')
        
        for column in object_dtype_list:
            self.df[column] = self.df[column].astype('string')
        print('---object 형식을 str 형식으로 변경 완료---')
        self.df['corp_estblsh_day'].unique()
        return self.df       
        
    
    def dt_data_handling(self):
        dt_list = ['STD_DATE_NEW','init_loan_dt','loan_deadln_dt']
        dt_list_2 = ['corp_estblsh_day','init_regist_dt','lst_renew_dt']
        
        for dt_col in dt_list:
            self.df[dt_col] = pd.to_datetime(self.df[dt_col].str.title(), format='%d%b%Y', errors='coerce') # Null 값은 그대로 놔둠
            print(f"{dt_col}이 날짜 형식으로 변환 완료되었습니다")
        for dt_col in dt_list_2:
            self.df[dt_col] = pd.to_datetime(self.df[dt_col], format='%d%b%Y',errors='coerce') # Null 값은 그대로 놔둠
            print(f"{dt_col}이 날짜 형식으로 변환 완료되었습니다")
        
        self.df['corp_estblsh_day'].fillna(pd.Timestamp('2024-01-01'), inplace=True)
        
        reference = pd.Timestamp('2024-01-01')
        print(reference)
        print(type(reference))
        
        self.df['cnt_days_since_0101'] = (self.df['STD_DATE_NEW'] - reference).dt.days
        self.df['cnt_days_from_init'] = (self.df['STD_DATE_NEW'] - self.df['init_loan_dt']).dt.days
        self.df['cnt_days_until_dead'] = (self.df['loan_deadln_dt'] - self.df['STD_DATE_NEW']).dt.days
        self.df['cnt_days_total'] = (self.df['loan_deadln_dt'] - self.df['init_loan_dt']).dt.days
        # raw_copied['cnt_days_estblsh'] = (self.df['STD_DATE_NEW'] - self.df['corp_estblsh_day']).dt.days
        self.df['cnt_days_regist'] = (self.df['STD_DATE_NEW'] - self.df['init_regist_dt']).dt.days
        self.df['cnt_days_renew'] = (self.df['STD_DATE_NEW'] - self.df['lst_renew_dt']).dt.days
        return self.df

    def data_shrinkage(self):
        self.df.drop(columns=dt_list,inplace=True)
        self.df.drop(columns=['AF_CRG','AF_KIFRS_ADJ_NRML','AF_KIFRS_ADJ_SM','AF_KIFRS_ADJ_FXD_UNDER','AF_NRML_RATIO',\
                                 'AF_SM_RATIO','AF_FXD_UNDER_RATIO','AF_ASSET_QUALITY','AF_DLNQNCY_DAY_CNT','AF_BNKRUT_CORP_FLG',\
                                'AF_BOND_ADJUST_FLG','AF_NO_PRFT_LOAN_MK','AF_NO_PRFT_LOAN_CAND_FLG','QUAL_CHG','NRML_CHG','SM_CHG','FXD_UNDER_CHG'],inplace=True)
        self.df.drop(columns=['BF_CRG','BF_KIFRS_ADJ_NRML','BF_KIFRS_ADJ_SM','BF_KIFRS_ADJ_FXD_UNDER','BF_NRML_RATIO',\
                                 'BF_SM_RATIO','BF_FXD_UNDER_RATIO','BF_ASSET_QUALITY','BF_DLNQNCY_DAY_CNT','BF_BNKRUT_CORP_FLG',\
                                'BF_BOND_ADJUST_FLG','BF_NO_PRFT_LOAN_MK','BF_NO_PRFT_LOAN_CAND_FLG'],inplace=True)
        return self.df
#여기까지 검토 완료
        
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
