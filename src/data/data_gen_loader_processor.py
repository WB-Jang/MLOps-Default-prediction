"""Data loading utilities for the MLOps pipeline."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict
from loguru import logger
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch

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
        self.scaled_df: Optional[pd.DataFrame] = None
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

    def detecting_type_encoding(self):
        num_col_list = self.df.select_dtypes(include=['float64','int64']).columns.tolist()
        str_col_list = self.df.select_dtypes(include='string').columns.tolist()
        
        str_col_list_less_target = []
        for col in str_col_list:
            if col != 'DLNQ_1M_FLG':
                str_col_list_less_target.append(col)
        
        print(f"수치형 컬럼은 총 {len(num_col_list)}개 입니다")
        print(f"범주형 컬럼은 총 {len(str_col_list_less_target)}개 입니다")

        raw_labeled_v1=self.df.copy()
        
        for col in str_col_list:
            raw_labeled_v1[col][raw_labeled_v1[col].isnull()] = 'NA'
        
        raw_labeled_v1['biz_gb'] = raw_labeled_v1['biz_gb'].str[:3]
        print(raw_labeled_v1['biz_gb'].unique())
        
        raw_labeled_v1['biz_gb'] = raw_labeled_v1['biz_gb'].str.lower()
        for col in str_col_list:
            print(raw_labeled_v1[col].unique())
        
        
        # LabelEncoder를 보관할 딕셔너리 (각 컬럼별로 학습된 encoder를 저장)
        raw_labeled = raw_labeled_v1.copy()
        encoders = {}
        # 각 범주형 컬럼에 대해 LabelEncoder로 변환
        for col in str_col_list:
            le = LabelEncoder()
            raw_labeled[col] = le.fit_transform(raw_labeled_v1[col])
            encoders[col] = le
        
        nunique_str = {}
        for i, c in enumerate(str_col_list_less_target):
            num_str = raw_labeled[c].max() + 1  # 라벨 인코딩 이후 각 컬럼별 value들 중 최대값 + 1 
            nunique_str[i] = num_str
        print(nunique_str)

        # 예시: 평균 대체
        #imputer = SimpleImputer(strategy="median") # "mean", "median", "most_frequent", "constant" 옵션 존재
        imputer = SimpleImputer(strategy="most_frequent")
        # fit_transform으로 결측치 보간
        num_data = raw_labeled[num_col_list].values 
        
        num_data_imputed = imputer.fit_transform(num_data)
        raw_labeled[num_col_list] = num_data_imputed
        print('---결측치 최빈값으로 보간 완료---')
        import pandas as pd
        from sklearn.preprocessing import RobustScaler
        
        # 로버스트 스케일러 초기화 (중앙값과 IQR을 사용)
        scaler = RobustScaler()
        
        raw_scaled = raw_labeled.copy()
        raw_scaled[num_col_list] = scaler.fit_transform(raw_labeled[num_col_list])
        self.scaled_df = raw_scaled
        print('---Robust Scaling completed---')
        return self.scaled_df
        
    def save_processed_data(self, output_path: str) -> None:
        """
        Save processed data to CSV.
        
        Args:
            output_path: Path to save the processed data
        """
        if self.scaled_df is None:
            raise RuntimeError("Data not loaded. Call load_raw_data() first.")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.scaled_df.to_csv(output_path, index=False)
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
    train_df, temp_df = train_test_split(
        raw_scaled,
        test_size=test_size,
        random_state=random_state,
        stratify=raw_scaled["DLNQ_1M_FLG"]
    )
        
    # Step 2: temp → fine_tune(15%) + test(15%) 분할
    fine_df, test_df = train_test_split(
        temp_df,
        fine_size=fine_size,  # 50% of 20% = 10% of original
        random_state=42,
        stratify=temp_df["DLNQ_1M_FLG"]
    )
                   
    X_train_str = train_df[str_col_list_less_target].values  # (행, 50)
    X_train_num = train_df[num_col_list].values  # (행, 45)
    y_train = train_df["DLNQ_1M_FLG"].values
    
    X_fine_str = fine_df[str_col_list_less_target].values
    X_fine_num = fine_df[num_col_list].values
    y_fine = fine_df["DLNQ_1M_FLG"].values
    
    X_test_str = test_df[str_col_list_less_target].values
    X_test_num = test_df[num_col_list].values
    y_test = test_df["DLNQ_1M_FLG"].values
    
    print("Train size:", len(train_df), "Fine size:", len(fine_df),"Test size:", len(test_df))
            
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
    return X_train_str, X_train_num, y_train, X_fine_str, X_fine_num, y_fine, X_test_str, X_test_num, y_test, metadata
