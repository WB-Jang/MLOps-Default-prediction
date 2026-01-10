"""
Generate raw synthetic data for loan default prediction.

This script creates synthetic loan application data that will be used as the 
raw data source for the entire MLOps pipeline. It generates realistic loan 
features including both categorical and numerical variables.

Usage:
    python generate_raw_data.py --num-rows 10000 --output-dir ./data/raw
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
from loguru import logger


def generate_categorical_features(num_rows: int) -> pd.DataFrame:
    """
    Generate categorical features for loan applications.
    
    Args:
        num_rows: Number of rows to generate
        
    Returns:
        DataFrame with categorical features
    """
    np.random.seed(42)
    
    categorical_data = {
        # Employment and Income
        'employment_type': np.random.choice(['Full-time', 'Part-time', 'Self-employed', 'Unemployed'], 
                                           size=num_rows, p=[0.6, 0.15, 0.2, 0.05]),
        'income_category': np.random.choice(['Low', 'Medium', 'High', 'Very High'], 
                                           size=num_rows, p=[0.2, 0.4, 0.3, 0.1]),
        'education_level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 
                                           size=num_rows, p=[0.3, 0.4, 0.25, 0.05]),
        
        # Credit History
        'credit_score_category': np.random.choice(['Poor', 'Fair', 'Good', 'Excellent'], 
                                                 size=num_rows, p=[0.15, 0.25, 0.4, 0.2]),
        'payment_history': np.random.choice(['No History', 'Some Delays', 'Good', 'Excellent'], 
                                           size=num_rows, p=[0.1, 0.2, 0.5, 0.2]),
        
        # Loan Details
        'loan_purpose': np.random.choice(['Home', 'Car', 'Education', 'Business', 'Personal', 'Debt Consolidation'], 
                                        size=num_rows, p=[0.25, 0.2, 0.15, 0.15, 0.15, 0.1]),
        'property_type': np.random.choice(['Apartment', 'House', 'Condo', 'No Property'], 
                                         size=num_rows, p=[0.3, 0.35, 0.2, 0.15]),
        
        # Demographics
        'marital_status': np.random.choice(['Single', 'Married', 'Divorced', 'Widowed'], 
                                          size=num_rows, p=[0.3, 0.5, 0.15, 0.05]),
        'dependents': np.random.choice(['0', '1', '2', '3+'], 
                                      size=num_rows, p=[0.3, 0.25, 0.25, 0.2]),
        'region': np.random.choice(['Urban', 'Suburban', 'Rural'], 
                                  size=num_rows, p=[0.5, 0.35, 0.15]),
    }
    
    return pd.DataFrame(categorical_data)


def generate_numerical_features(num_rows: int) -> pd.DataFrame:
    """
    Generate numerical features for loan applications.
    
    Args:
        num_rows: Number of rows to generate
        
    Returns:
        DataFrame with numerical features
    """
    np.random.seed(42)
    
    numerical_data = {
        # Financial Metrics
        'annual_income': np.random.lognormal(10.5, 0.8, num_rows).clip(15000, 500000),
        'loan_amount': np.random.lognormal(10, 1, num_rows).clip(1000, 100000),
        'debt_to_income_ratio': np.random.beta(2, 5, num_rows) * 100,
        'credit_score': np.random.normal(680, 80, num_rows).clip(300, 850),
        'existing_debt': np.random.lognormal(9, 1.5, num_rows).clip(0, 200000),
        
        # Employment History
        'employment_length_years': np.random.exponential(5, num_rows).clip(0, 40),
        'months_at_current_job': np.random.exponential(36, num_rows).clip(0, 480),
        
        # Loan Terms
        'loan_term_months': np.random.choice([12, 24, 36, 48, 60, 84, 120, 180, 240, 360], 
                                            size=num_rows, p=[0.05, 0.1, 0.15, 0.15, 0.2, 0.1, 0.1, 0.05, 0.05, 0.05]),
        'interest_rate': np.random.gamma(2, 2, num_rows).clip(2, 30),
        
        # Account Information
        'num_credit_lines': np.random.poisson(3, num_rows).clip(0, 20),
        'num_credit_inquiries': np.random.poisson(1.5, num_rows).clip(0, 15),
        'revolving_balance': np.random.lognormal(7, 2, num_rows).clip(0, 100000),
        'revolving_utilization': np.random.beta(2, 3, num_rows) * 100,
        
        # Personal Information
        'age': np.random.normal(40, 12, num_rows).clip(18, 80),
        'months_since_last_delinquency': np.random.exponential(24, num_rows).clip(0, 120),
    }
    
    return pd.DataFrame(numerical_data)


def generate_target_variable(df: pd.DataFrame) -> pd.Series:
    """
    Generate target variable (default) based on features with realistic patterns.
    
    Args:
        df: DataFrame with features
        
    Returns:
        Series with target labels (0 = no default, 1 = default)
    """
    np.random.seed(42)
    
    # Calculate default probability based on risk factors
    risk_score = np.zeros(len(df))
    
    # Credit score impact (stronger predictor)
    if 'credit_score' in df.columns:
        risk_score += (850 - df['credit_score']) / 100
    
    # Debt to income ratio impact
    if 'debt_to_income_ratio' in df.columns:
        risk_score += df['debt_to_income_ratio'] / 50
    
    # Employment type impact
    if 'employment_type' in df.columns:
        employment_risk = {'Full-time': 0, 'Part-time': 0.5, 'Self-employed': 0.3, 'Unemployed': 2}
        risk_score += df['employment_type'].map(employment_risk)
    
    # Income impact
    if 'annual_income' in df.columns:
        risk_score += (100000 - df['annual_income']) / 50000
    
    # Credit score category impact
    if 'credit_score_category' in df.columns:
        credit_risk = {'Excellent': 0, 'Good': 0.3, 'Fair': 0.8, 'Poor': 1.5}
        risk_score += df['credit_score_category'].map(credit_risk)
    
    # Normalize risk score to probability
    default_probability = 1 / (1 + np.exp(-risk_score + 3))  # Sigmoid with offset
    
    # Generate binary outcome based on probability
    defaults = (np.random.random(len(df)) < default_probability).astype(int)
    
    return defaults


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features as integers for model training.
    
    Args:
        df: DataFrame with categorical features
        
    Returns:
        DataFrame with encoded features
    """
    encoded_df = df.copy()
    
    for col in df.columns:
        if df[col].dtype == 'object':
            encoded_df[col] = pd.Categorical(df[col]).codes
    
    return encoded_df


def generate_loan_default_dataset(
    num_rows: int = 10000,
    include_target: bool = True,
    encode_categoricals: bool = False
) -> pd.DataFrame:
    """
    Generate complete loan default prediction dataset.
    
    Args:
        num_rows: Number of loan applications to generate
        include_target: Whether to include the target variable
        encode_categoricals: Whether to encode categorical variables
        
    Returns:
        DataFrame with complete dataset
    """
    logger.info(f"Generating {num_rows} loan application records...")
    
    # Generate features
    cat_features = generate_categorical_features(num_rows)
    num_features = generate_numerical_features(num_rows)
    
    # Combine features
    df = pd.concat([cat_features, num_features], axis=1)
    
    # Add identifier
    df.insert(0, 'loan_id', [f'LOAN-{str(i).zfill(8)}' for i in range(num_rows)])
    
    # Generate target if requested
    if include_target:
        df['default'] = generate_target_variable(df)
        logger.info(f"Default rate: {df['default'].mean():.2%}")
    
    # Encode categoricals if requested
    if encode_categoricals:
        # Keep loan_id and default unchanged
        cols_to_encode = [col for col in cat_features.columns]
        encoded_features = encode_categorical_features(df[cols_to_encode])
        df[cols_to_encode] = encoded_features
    
    return df


def main():
    """Main function to generate and save raw data."""
    parser = argparse.ArgumentParser(
        description='Generate synthetic raw data for loan default prediction'
    )
    parser.add_argument(
        '--num-rows',
        type=int,
        default=10000,
        help='Number of loan applications to generate (default: 10000)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./data/raw',
        help='Output directory for raw data (default: ./data/raw)'
    )
    parser.add_argument(
        '--filename',
        type=str,
        default='synthetic_data.csv',
        help='Output filename (default: synthetic_data.csv)'
    )
    parser.add_argument(
        '--no-target',
        action='store_true',
        help='Generate data without target variable'
    )
    parser.add_argument(
        '--encoded',
        action='store_true',
        help='Encode categorical variables as integers'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger.add(
        f"logs/data_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        rotation="500 MB"
    )
    
    logger.info("=" * 60)
    logger.info("Loan Default Prediction - Raw Data Generation")
    logger.info("=" * 60)
    logger.info(f"Configuration:")
    logger.info(f"  - Number of rows: {args.num_rows:,}")
    logger.info(f"  - Output directory: {args.output_dir}")
    logger.info(f"  - Filename: {args.filename}")
    logger.info(f"  - Include target: {not args.no_target}")
    logger.info(f"  - Encode categoricals: {args.encoded}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")
    
    # Generate dataset
    df = generate_loan_default_dataset(
        num_rows=args.num_rows,
        include_target=not args.no_target,
        encode_categoricals=args.encoded
    )
    
    # Save to CSV
    output_path = output_dir / args.filename
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    logger.info(f"Saved dataset to: {output_path}")
    
    # Print dataset info
    logger.info(f"\nDataset shape: {df.shape}")
    logger.info(f"Columns ({len(df.columns)}): {', '.join(df.columns.tolist())}")
    logger.info(f"\nFirst few rows:")
    logger.info(f"\n{df.head()}")
    
    # Save data info
    info_path = output_dir / f"{args.filename.replace('.csv', '_info.txt')}"
    with open(info_path, 'w') as f:
        f.write("Dataset Information\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Rows: {len(df):,}\n")
        f.write(f"Columns: {len(df.columns)}\n\n")
        f.write("Column Details:\n")
        f.write("-" * 60 + "\n")
        f.write(df.dtypes.to_string())
        f.write("\n\n")
        f.write("Statistical Summary:\n")
        f.write("-" * 60 + "\n")
        f.write(df.describe().to_string())
    
    logger.info(f"Saved dataset info to: {info_path}")
    logger.info("\n" + "=" * 60)
    logger.info("Data generation completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
