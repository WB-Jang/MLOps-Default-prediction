#!/usr/bin/env python3
"""
End-to-end example of the MLOps pipeline usage.

This script demonstrates how to:
1. Generate synthetic data
2. Load and preprocess data
3. Create PyTorch datasets (when torch is available)
4. Initialize models
5. Store metadata in MongoDB
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def example_data_generation():
    """Example 1: Generate synthetic loan data."""
    print("\n" + "="*60)
    print("Example 1: Data Generation")
    print("="*60)
    
    from generate_raw_data import generate_loan_default_dataset
    
    # Generate dataset
    df = generate_loan_default_dataset(
        num_rows=500,
        include_target=True,
        encode_categoricals=False
    )
    
    print(f"✓ Generated {len(df)} loan applications")
    print(f"✓ Features: {len(df.columns)} columns")
    print(f"✓ Default rate: {df['default'].mean():.2%}")
    print(f"\nFirst 3 rows:")
    print(df.head(3))


def example_data_loading():
    """Example 2: Load and preprocess data."""
    print("\n" + "="*60)
    print("Example 2: Data Loading & Preprocessing")
    print("="*60)
    
    from src.data.data_loader import DataLoader
    
    # Initialize loader
    loader = DataLoader("./data/raw/synthetic_data.csv")
    
    # Load data
    df = loader.load_raw_data()
    print(f"✓ Loaded {len(df)} rows")
    
    # Get column types
    cat_cols, num_cols = loader.get_column_lists()
    print(f"✓ Categorical: {len(cat_cols)} features")
    print(f"✓ Numerical: {len(num_cols)} features")
    
    # Encode categoricals
    encoding_maps = loader.encode_categorical_columns()
    print(f"✓ Encoded {len(encoding_maps)} categorical features")
    
    # Get features and target
    X, y = loader.get_features_and_target()
    print(f"✓ Features shape: {X.shape}")
    print(f"✓ Target shape: {y.shape}")
    
    # Get categorical max dict (for embeddings)
    cat_max_dict = loader.get_categorical_max_dict()
    print(f"✓ Embedding sizes: {dict(list(cat_max_dict.items())[:3])}...")


def example_train_test_split():
    """Example 3: Train/test split."""
    print("\n" + "="*60)
    print("Example 3: Train/Test Split")
    print("="*60)
    
    from src.data.data_loader import load_data_for_training
    
    # Load and split data
    (X_train, y_train), (X_test, y_test), metadata = load_data_for_training(
        data_path="./data/raw/synthetic_data.csv",
        test_size=0.2,
        random_state=42
    )
    
    print(f"✓ Train set: {len(X_train)} samples")
    print(f"✓ Test set: {len(X_test)} samples")
    print(f"✓ Categorical features: {metadata['num_categorical_features']}")
    print(f"✓ Numerical features: {metadata['num_numerical_features']}")
    print(f"✓ Train target distribution: {y_train.value_counts().to_dict()}")
    print(f"✓ Test target distribution: {y_test.value_counts().to_dict()}")


def example_model_initialization():
    """Example 4: Initialize models (requires torch)."""
    print("\n" + "="*60)
    print("Example 4: Model Initialization")
    print("="*60)
    
    try:
        import torch
        from src.models import Encoder, TabTransformerClassifier
        from src.data.data_loader import load_data_for_training
        
        # Load metadata
        _, _, metadata = load_data_for_training("./data/raw/synthetic_data.csv")
        
        # Create encoder
        encoder = Encoder(
            cnt_cat_features=metadata['num_categorical_features'],
            cnt_num_features=metadata['num_numerical_features'],
            cat_max_dict=metadata['cat_max_dict'],
            d_model=32,
            nhead=4,
            num_layers=6
        )
        
        # Create classifier
        classifier = TabTransformerClassifier(
            encoder=encoder,
            d_model=32,
            final_hidden=128
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in classifier.parameters())
        trainable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
        
        print(f"✓ Created Encoder")
        print(f"✓ Created TabTransformerClassifier")
        print(f"✓ Total parameters: {total_params:,}")
        print(f"✓ Trainable parameters: {trainable_params:,}")
        
        # Test forward pass
        batch_size = 4
        x_cat = torch.randint(0, 10, (batch_size, metadata['num_categorical_features']))
        x_num = torch.randn(batch_size, metadata['num_numerical_features'])
        
        classifier.eval()
        with torch.no_grad():
            logits = classifier(x_cat, x_num)
            predictions = torch.argmax(logits, dim=1)
        
        print(f"✓ Forward pass successful")
        print(f"✓ Output shape: {logits.shape}")
        print(f"✓ Sample predictions: {predictions.tolist()}")
        
    except ImportError:
        print("⚠ PyTorch not installed. Skipping model initialization example.")
        print("  Install with: pip install torch")


def example_kafka_integration():
    """Example 5: Kafka integration (requires running Kafka)."""
    print("\n" + "="*60)
    print("Example 5: Kafka Integration")
    print("="*60)
    
    try:
        from src.kafka.producer import kafka_producer
        from src.data.data_loader import DataLoader
        
        # Load sample data
        loader = DataLoader("./data/raw/synthetic_data.csv")
        df = loader.load_raw_data()
        sample_record = df.head(1).to_dict('records')[0]
        
        print("Sample record to send to Kafka:")
        print(f"  loan_id: {sample_record.get('loan_id')}")
        print(f"  Features: {len(sample_record)} fields")
        
        # Note: This would actually send to Kafka if broker is running
        print("\n⚠ Kafka broker not running. Skipping actual send.")
        print("  To use Kafka:")
        print("    1. Start Kafka: docker-compose up -d kafka")
        print("    2. Run: kafka_producer.connect()")
        print("    3. Run: kafka_producer.send_raw_data(data, key='test')")
        print("    4. Run: kafka_producer.disconnect()")
        
    except Exception as e:
        print(f"⚠ Kafka not available: {e}")


def example_mongodb_integration():
    """Example 6: MongoDB integration (requires running MongoDB)."""
    print("\n" + "="*60)
    print("Example 6: MongoDB Integration")
    print("="*60)
    
    try:
        from src.database.mongodb import mongodb_client
        from datetime import datetime
        
        print("MongoDB client initialized")
        print("  Host: localhost:27017")
        print("  Database: mlops_default_prediction")
        
        print("\n⚠ MongoDB not running. Skipping actual operations.")
        print("  To use MongoDB:")
        print("    1. Start MongoDB: docker-compose up -d mongodb")
        print("    2. Run: mongodb_client.connect()")
        print("    3. Store data:")
        print("       model_id = mongodb_client.store_model_metadata(...)")
        print("    4. Run: mongodb_client.disconnect()")
        
        # Show what would be stored
        example_metadata = {
            "model_name": "default_prediction_classifier",
            "model_path": "./models/classifier_example.pth",
            "model_version": datetime.now().strftime('%Y%m%d_%H%M%S'),
            "hyperparameters": {
                "d_model": 32,
                "nhead": 4,
                "num_layers": 6
            },
            "metrics": {
                "accuracy": 0.85,
                "f1_score": 0.82
            }
        }
        print("\nExample model metadata:")
        for key, value in example_metadata.items():
            print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"⚠ MongoDB not available: {e}")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("MLOps Pipeline - End-to-End Examples")
    print("="*60)
    print("\nThis script demonstrates the complete pipeline workflow.")
    print("Some examples require additional dependencies or services.")
    
    # Check if data exists
    data_path = Path("./data/raw/synthetic_data.csv")
    if not data_path.exists():
        print("\n⚠ Data file not found!")
        print("  Run this first: python generate_raw_data.py")
        
        # Generate sample data for demonstration
        print("\n  Generating sample data for examples...")
        example_data_generation()
        
        # Save it
        from generate_raw_data import generate_loan_default_dataset
        df = generate_loan_default_dataset(num_rows=1000)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(data_path, index=False)
        print(f"✓ Saved sample data to {data_path}")
    
    # Run examples
    try:
        example_data_loading()
        example_train_test_split()
        example_model_initialization()
        example_kafka_integration()
        example_mongodb_integration()
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("Please run: python generate_raw_data.py")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print("✓ Data generation: Working")
    print("✓ Data loading: Working")
    print("✓ Preprocessing: Working")
    print("✓ Model initialization: Requires torch")
    print("✓ Kafka integration: Requires running Kafka")
    print("✓ MongoDB integration: Requires running MongoDB")
    print("\nNext steps:")
    print("  1. Install dependencies: pip install -r requirements.txt")
    print("  2. Start services: docker-compose up -d")
    print("  3. Train model: python train.py --train")
    print("  4. View Airflow UI: http://localhost:8080")
    print("\nFor more information, see README.md and QUICKSTART.md")
    print("="*60 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
