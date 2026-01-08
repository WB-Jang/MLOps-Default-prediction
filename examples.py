"""Example usage of the MLOps pipeline components."""
import torch
from src.models import Encoder, TabTransformerClassifier
from src.kafka.producer import kafka_producer
from src.kafka.consumer import RawDataConsumer
from src.database.mongodb import mongodb_client
from config.settings import settings


def example_kafka_usage():
    """Example: Send and receive data through Kafka."""
    print("=== Kafka Example ===")
    
    # Connect to Kafka producer
    kafka_producer.connect()
    
    # Send sample data
    sample_data = {
        "loan_id": "LOAN-001",
        "categorical_features": [1, 2, 3, 4, 5],
        "numerical_features": [1000.0, 2.5, 750.0, 30.0, 15000.0]
    }
    
    kafka_producer.send_raw_data(sample_data, key="loan-001")
    print(f"Sent data to Kafka: {sample_data}")
    
    kafka_producer.disconnect()
    
    # Note: To consume data, you would create a consumer:
    # consumer = RawDataConsumer()
    # consumer.connect()
    # messages = consumer.consume_batch(batch_size=10)
    # consumer.disconnect()


def example_mongodb_usage():
    """Example: Store and retrieve data from MongoDB."""
    print("\n=== MongoDB Example ===")
    
    # Connect to MongoDB
    mongodb_client.connect()
    
    # Store model metadata
    model_id = mongodb_client.store_model_metadata(
        model_name="example_classifier",
        model_path="/path/to/model.pth",
        model_version="v1.0.0",
        hyperparameters={
            "d_model": 32,
            "nhead": 4,
            "num_layers": 6
        },
        metrics={
            "accuracy": 0.85,
            "f1_score": 0.82
        }
    )
    print(f"Stored model metadata with ID: {model_id}")
    
    # Retrieve model metadata
    metadata = mongodb_client.get_model_metadata("example_classifier")
    print(f"Retrieved metadata: {metadata['model_name']} v{metadata['model_version']}")
    
    # Store a prediction
    pred_id = mongodb_client.store_prediction(
        model_id=model_id,
        input_data={"feature1": 1.0, "feature2": 2.0},
        prediction=0,
        confidence=0.95
    )
    print(f"Stored prediction with ID: {pred_id}")
    
    mongodb_client.disconnect()


def example_model_usage():
    """Example: Create and use the model."""
    print("\n=== Model Example ===")
    
    # Create encoder
    cat_max_dict = {i: 10 for i in range(5)}
    encoder = Encoder(
        cnt_cat_features=5,
        cnt_num_features=5,
        cat_max_dict=cat_max_dict,
        d_model=settings.d_model,
        nhead=settings.nhead,
        num_layers=settings.num_layers
    )
    
    # Create classifier
    classifier = TabTransformerClassifier(
        encoder=encoder,
        d_model=settings.d_model
    )
    
    print(f"Created classifier with {sum(p.numel() for p in classifier.parameters())} parameters")
    
    # Example forward pass
    batch_size = 4
    x_cat = torch.randint(0, 10, (batch_size, 5))
    x_num = torch.randn(batch_size, 5)
    
    classifier.eval()
    with torch.no_grad():
        logits = classifier(x_cat, x_num)
        predictions = torch.argmax(logits, dim=1)
    
    print(f"Example predictions: {predictions.tolist()}")
    
    # Save model
    model_path = "example_model.pth"
    torch.save({
        'model_state_dict': classifier.state_dict(),
        'hyperparameters': {
            'd_model': settings.d_model,
            'nhead': settings.nhead,
            'num_layers': settings.num_layers
        }
    }, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    print("MLOps Pipeline Examples")
    print("=" * 50)
    
    # Note: These examples require Kafka and MongoDB to be running
    # Start services with: docker-compose up -d
    
    try:
        # example_kafka_usage()
        # example_mongodb_usage()
        example_model_usage()
        
        print("\n" + "=" * 50)
        print("Examples completed successfully!")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure Kafka and MongoDB are running:")
        print("  docker-compose up -d")
