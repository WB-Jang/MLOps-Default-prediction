"""Data loading utilities for the MLOps pipeline."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict
from loguru import logger
from sklearn.preprocessing import LabelEncoder, RobustScaler
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
from sdv.single_table import GaussianCopulaSynthesizer

class DataGenLoaderProcessor:
    """Generate and Load and manage raw data for the pipeline."""
    
    def __init__(self, data_path: str = "./data/raw/synthetic_data.csv"):
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
        import os
        model_path = os.path.join(os.path.dirname(__file__), 'distribution_model.pkl')
        synthesizer = GaussianCopulaSynthesizer.load(model_path)
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
        if self.df is None:
            raise RuntimeError(
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
        if self.df is None:
            raise RuntimeError(
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
        dt_list = ['STD_DATE_NEW','init_loan_dt','loan_deadln_dt']
        self.df.drop(columns=dt_list,inplace=True)
        self.df.drop(columns=['AF_CRG','AF_KIFRS_ADJ_NRML','AF_KIFRS_ADJ_SM','AF_KIFRS_ADJ_FXD_UNDER','AF_NRML_RATIO',\
                                 'AF_SM_RATIO','AF_FXD_UNDER_RATIO','AF_ASSET_QUALITY','AF_DLNQNCY_DAY_CNT','AF_BNKRUT_CORP_FLG',\
                                'AF_BOND_ADJUST_FLG','AF_NO_PRFT_LOAN_MK','AF_NO_PRFT_LOAN_CAND_FLG','QUAL_CHG','NRML_CHG','SM_CHG','FXD_UNDER_CHG'],inplace=True)
        self.df.drop(columns=['BF_CRG','BF_KIFRS_ADJ_NRML','BF_KIFRS_ADJ_SM','BF_KIFRS_ADJ_FXD_UNDER','BF_NRML_RATIO',\
                                 'BF_SM_RATIO','BF_FXD_UNDER_RATIO','BF_ASSET_QUALITY','BF_DLNQNCY_DAY_CNT','BF_BNKRUT_CORP_FLG',\
                                'BF_BOND_ADJUST_FLG','BF_NO_PRFT_LOAN_MK','BF_NO_PRFT_LOAN_CAND_FLG'],inplace=True)
        return self.df

    def detecting_type_encoding(self):
        # DataFrame에서 다시 컬럼 타입을 확인 (data_shrinkage 이후)
        num_col_list = self.df. select_dtypes(include=['float64','int64']).columns.tolist()
        str_col_list = self.df.select_dtypes(include='string').columns.tolist()
        
        str_col_list_less_target = []
        for col in str_col_list:
            if col != 'DLNQ_1M_FLG':  
                str_col_list_less_target.append(col)
        
        print(f"수치형 컬럼은 총 {len(num_col_list)}개 입니다")
        print(f"범주형 컬럼은 총 {len(str_col_list_less_target)}개 입니다")

        raw_labeled_v1 = self.df.copy()
        
        # Null 값 처리
        for col in str_col_list:  
            raw_labeled_v1.loc[raw_labeled_v1[col].isnull(), col] = 'NA'
        
        raw_labeled_v1['biz_gb'] = raw_labeled_v1['biz_gb'].str[: 3]
        print(raw_labeled_v1['biz_gb'].unique())
        
        raw_labeled_v1['biz_gb'] = raw_labeled_v1['biz_gb']. str.lower()
        for col in str_col_list: 
            print(raw_labeled_v1[col]. unique())
        
        # LabelEncoder를 보관할 딕셔너리
        raw_labeled = raw_labeled_v1.copy()
        encoders = {}
        
        # 각 범주형 컬럼에 대해 LabelEncoder로 변환
        for col in str_col_list: 
            le = LabelEncoder()
            raw_labeled[col] = le. fit_transform(raw_labeled_v1[col])
            encoders[col] = le
        
        nunique_str = {}
        for i, c in enumerate(str_col_list_less_target):
            num_str = raw_labeled[c].max() + 1
            nunique_str[i] = num_str
        print(nunique_str)

        # ⭐ LabelEncoding 후 수치형 컬럼 다시 계산
        num_col_list = raw_labeled.select_dtypes(include=['float64','int64','int32','int8']).columns.tolist()
        print(f"LabelEncoding 후 실제 수치형 컬럼:  {len(num_col_list)}개")

        # ⭐ 완전 결측 컬럼 제거
        valid_num_cols = []
        for col in num_col_list:
            if raw_labeled[col].notna().sum() > 0:
                valid_num_cols.append(col)
            else:
                print(f"⚠️ 완전 결측 컬럼 제거: {col}")

        num_col_list = valid_num_cols
        print(f"유효한 수치형 컬럼:  {len(num_col_list)}개")
        # 결측치 보간
        imputer = SimpleImputer(strategy="most_frequent")
        num_data = raw_labeled[num_col_list].values
        num_data_imputed = imputer.fit_transform(num_data)
        
        # DataFrame으로 변환하여 할당
        import pandas as pd
        raw_labeled[num_col_list] = pd.DataFrame(
            num_data_imputed, 
            columns=num_col_list,
            index=raw_labeled. index
        )
        print('---결측치 최빈값으로 보간 완료---')
        
        # 로버스트 스케일러
        scaler = RobustScaler()
        raw_scaled = raw_labeled.copy()
        num_scaled = scaler.fit_transform(raw_labeled[num_col_list])
        raw_scaled[num_col_list] = pd.DataFrame(
            num_scaled,
            columns=num_col_list,
            index=raw_scaled.index
        )
        self.scaled_df = raw_scaled
        print('---Robust Scaling completed---')
        
        return self.scaled_df, str_col_list_less_target, num_col_list, nunique_str

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
def load_data_for_training(data_path: str ="./data/raw/synthetic_data.csv", test_size: float = 0.3, random_state: int = 42):
    """
    Load and preprocess data for training, ensuring Target is Numeric.
    """
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder, RobustScaler
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    import numpy as np
    
    logger.info(f"Loading raw data from {data_path}")
    try:
        raw_data = pd.read_csv(data_path, sep=',', encoding='euc-kr')
    except UnicodeDecodeError:
        raw_data = pd.read_csv(data_path, sep=',')
        
    logger.info(f"Loaded {len(raw_data)} rows")

    # [Target Name Correction]
    if 'target' in raw_data.columns and 'DLNQ_1M_FLG' not in raw_data.columns:
         raw_data.rename(columns={'target': 'DLNQ_1M_FLG'}, inplace=True)
    
    raw_copied = raw_data.copy()

    # 1. Null Check & Column Creation
    null_columns = raw_copied.columns[raw_copied.isnull().any()]
    for col in null_columns:
        new_column = f'{col}_isnull'
        raw_copied[new_column] = raw_copied[col].isnull().map(lambda x: 'Null' if x else 'Notnull')
    
    # 2. Object to String
    object_dtype_list = raw_copied.select_dtypes(include='object').columns.tolist()
    if 'acct_titl_cd' in raw_copied.columns and 'acct_titl_cd' not in object_dtype_list:
        object_dtype_list.append('acct_titl_cd')
    
    for column in object_dtype_list:
        raw_copied[column] = raw_copied[column].astype('string')
        
    # [Early Drops]
    early_drops = ['SSN_CORPNO','ACCT_KEY','acct_num','std_date','spcl_corp_new_mk']
    real_early_drops = [c for c in early_drops if c in raw_copied.columns]
    raw_copied.drop(columns=real_early_drops, inplace=True)
    logger.info(f"✅ Dropped ID columns: {real_early_drops}")

    # 3. Date Handling
    dt_list = ['STD_DATE_NEW','init_loan_dt','loan_deadln_dt']
    dt_list_2 = ['corp_estblsh_day','init_regist_dt','lst_renew_dt']
    
    existing_dt_list = [c for c in dt_list if c in raw_copied.columns]
    for dt_col in existing_dt_list:
        raw_copied[dt_col] = pd.to_datetime(raw_copied[dt_col].astype(str).str.title(), format='%d%b%Y', errors='coerce')

    existing_dt_list_2 = [c for c in dt_list_2 if c in raw_copied.columns]
    for dt_col in existing_dt_list_2:
        raw_copied[dt_col] = pd.to_datetime(raw_copied[dt_col], format='%d%b%Y', errors='coerce')

    if 'corp_estblsh_day' in raw_copied.columns:
        raw_copied['corp_estblsh_day'].fillna(pd.Timestamp('2024-01-01'), inplace=True)

    reference = pd.Timestamp('2024-01-01')
    
    if 'STD_DATE_NEW' in raw_copied.columns:
        raw_copied['cnt_days_since_0101'] = (raw_copied['STD_DATE_NEW'] - reference).dt.days
        if 'init_loan_dt' in raw_copied.columns:
            raw_copied['cnt_days_from_init'] = (raw_copied['STD_DATE_NEW'] - raw_copied['init_loan_dt']).dt.days
        if 'loan_deadln_dt' in raw_copied.columns:
            raw_copied['cnt_days_until_dead'] = (raw_copied['loan_deadln_dt'] - raw_copied['STD_DATE_NEW']).dt.days
        if 'loan_deadln_dt' in raw_copied.columns and 'init_loan_dt' in raw_copied.columns:
            raw_copied['cnt_days_total'] = (raw_copied['loan_deadln_dt'] - raw_copied['init_loan_dt']).dt.days
        if 'init_regist_dt' in raw_copied.columns:
            raw_copied['cnt_days_regist'] = (raw_copied['STD_DATE_NEW'] - raw_copied['init_regist_dt']).dt.days
        if 'lst_renew_dt' in raw_copied.columns:
            raw_copied['cnt_days_renew'] = (raw_copied['STD_DATE_NEW'] - raw_copied['lst_renew_dt']).dt.days

    # 4. Data Shrinkage
    raw_copied.drop(columns=existing_dt_list, inplace=True)
    
    drop_list_1 = ['AF_CRG','AF_KIFRS_ADJ_NRML','AF_KIFRS_ADJ_SM','AF_KIFRS_ADJ_FXD_UNDER','AF_NRML_RATIO',
                   'AF_SM_RATIO','AF_FXD_UNDER_RATIO','AF_ASSET_QUALITY','AF_DLNQNCY_DAY_CNT','AF_BNKRUT_CORP_FLG',
                   'AF_BOND_ADJUST_FLG','AF_NO_PRFT_LOAN_MK','AF_NO_PRFT_LOAN_CAND_FLG','QUAL_CHG','NRML_CHG','SM_CHG','FXD_UNDER_CHG']
    drop_list_2 = ['BF_CRG','BF_KIFRS_ADJ_NRML','BF_KIFRS_ADJ_SM','BF_KIFRS_ADJ_FXD_UNDER','BF_NRML_RATIO',
                   'BF_SM_RATIO','BF_FXD_UNDER_RATIO','BF_ASSET_QUALITY','BF_DLNQNCY_DAY_CNT','BF_BNKRUT_CORP_FLG',
                   'BF_BOND_ADJUST_FLG','BF_NO_PRFT_LOAN_MK','BF_NO_PRFT_LOAN_CAND_FLG']
    
    all_drops = drop_list_1 + drop_list_2
    real_drops = [c for c in all_drops if c in raw_copied.columns]
    raw_copied.drop(columns=real_drops, inplace=True)
    logger.info(f"✅ Dropped unused features: {len(real_drops)} columns")

    # 5. Type Separation
    num_col_list = raw_copied.select_dtypes(include=['float64','int64']).columns.tolist()
    str_col_list = raw_copied.select_dtypes(include='string').columns.tolist()
    
    target_col = 'DLNQ_1M_FLG'
    str_col_list_less_target = [col for col in str_col_list if col != target_col]
    
    # 6. Null Filling & Preprocessing
    raw_labeled_v1 = raw_copied.copy()
    for col in str_col_list:
        raw_labeled_v1[col] = raw_labeled_v1[col].fillna('NA')
        
    if 'biz_gb' in raw_labeled_v1.columns:
        raw_labeled_v1['biz_gb'] = raw_labeled_v1['biz_gb'].str[:3].str.lower()

    # 7. Label Encoding (Features)
    raw_labeled = raw_labeled_v1.copy()
    encoders = {}
    cat_max_dict = {}
    
    for i, col in enumerate(str_col_list_less_target):
        le = LabelEncoder()
        raw_labeled[col] = le.fit_transform(raw_labeled_v1[col])
        encoders[col] = le
        cat_max_dict[i] = int(raw_labeled[col].max() + 1)
        
    # ⭐ [CRITICAL FIX] Target Encoding (String -> Int)
    if target_col in raw_labeled.columns:
        # 데이터가 '0', '1' 문자열이거나 숫자형이 섞여있을 수 있음 -> 강제로 int 변환
        try:
            # 먼저 float으로 변환 후 int로 (예: '1.0' 스트링 대응)
            raw_labeled[target_col] = pd.to_numeric(raw_labeled[target_col], errors='coerce').fillna(0).astype(int)
            logger.info(f"✅ Converted Target '{target_col}' to integers.")
        except Exception as e:
            logger.warning(f"Target conversion failed, trying LabelEncoder: {e}")
            le_target = LabelEncoder()
            raw_labeled[target_col] = le_target.fit_transform(raw_labeled[target_col].astype(str))
            
    # 8. Numerical Imputation & Scaling
    if num_col_list:
        valid_num_cols = []
        dropped_null_cols = []
        for col in num_col_list:
             if raw_labeled[col].notna().any():
                 valid_num_cols.append(col)
             else:
                 dropped_null_cols.append(col)
                 raw_labeled.drop(columns=[col], inplace=True)
        
        if dropped_null_cols:
            logger.warning(f"⚠️ Dropped fully null numerical columns: {dropped_null_cols}")
        
        num_col_list = valid_num_cols 

        if num_col_list:
            imputer = SimpleImputer(strategy="most_frequent")
            raw_labeled[num_col_list] = imputer.fit_transform(raw_labeled[num_col_list])
            
            scaler = RobustScaler()
            raw_labeled[num_col_list] = scaler.fit_transform(raw_labeled[num_col_list])
    
    # 9. Split
    if target_col not in raw_labeled.columns:
        raise KeyError(f"Target column '{target_col}' not found!")

    train_df, temp_df = train_test_split(
        raw_labeled, test_size=0.3, random_state=random_state, stratify=raw_labeled[target_col]
    )
    fine_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=random_state, stratify=temp_df[target_col]
    )
    
    # 10. Extract
    X_train_str = train_df[str_col_list_less_target].values
    X_train_num = train_df[num_col_list].values
    y_train = train_df[target_col].values
    
    X_fine_str = fine_df[str_col_list_less_target].values
    X_fine_num = fine_df[num_col_list].values
    y_fine = fine_df[target_col].values
    
    X_test_str = test_df[str_col_list_less_target].values
    X_test_num = test_df[num_col_list].values
    y_test = test_df[target_col].values
    
    metadata = {
        'num_categorical_features': len(str_col_list_less_target),
        'num_numerical_features': len(num_col_list),
        'cat_max_dict': cat_max_dict,
        'categorical_columns': str_col_list_less_target,
        'numerical_columns': num_col_list,
        'encoders': encoders
    }
    
    return X_train_str, X_train_num, y_train, X_fine_str, X_fine_num, y_fine, X_test_str, X_test_num, y_test, metadata