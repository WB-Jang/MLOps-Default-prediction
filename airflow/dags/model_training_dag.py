"""Airflow DAG for model training."""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "model_training",
    default_args=default_args,
    description="Train loan default prediction model",
    schedule_interval=None,  # Triggered manually or by other DAGs
    catchup=False,
    tags=["model", "training"],
)


def prepare_training_data(**context):
    """Prepare data for model training."""
    import sys
    sys.path.insert(0, "/opt/airflow")
    
    from config.settings import settings
    from src.data import DatabaseManager
    
    db = DatabaseManager(settings.database_url)
    
    # Get new data for training
    new_data = db.get_new_data_for_training()
    print(f"Retrieved {len(new_data)} new records for training")
    
    # Store count in XCom for downstream tasks
    context["ti"].xcom_push(key="training_sample_count", value=len(new_data))
    
    return "success"


def pretrain_model(**context):
    """Pretrain model using contrastive learning."""
    import sys
    sys.path.insert(0, "/opt/airflow")
    
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    
    from config.settings import settings
    from src.models import Encoder, ProjectionHead
    from src.models.training import pretrain_contrastive
    
    print("Starting model pretraining...")
    
    # TODO: Load actual data from database
    # For now, this is a placeholder showing the structure
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Example: Create dummy data for demonstration
    # In production, load from database
    # cat_max_dict should be calculated from actual data
    cat_max_dict = {i: 10 for i in range(10)}  # Placeholder
    
    encoder = Encoder(
        cnt_cat_features=10,
        cnt_num_features=10,
        cat_max_dict=cat_max_dict,
        d_model=settings.d_model,
        nhead=settings.nhead,
        num_layers=settings.num_layers,
        dim_feedforward=settings.dim_feedforward,
        dropout_rate=settings.dropout_rate,
    )
    
    projection_head = ProjectionHead(
        d_model=settings.d_model,
        projection_dim=settings.projection_dim,
    )
    
    # TODO: Replace with actual data loader
    # pretrain_loader = DataLoader(...)
    # encoder, projection_head = pretrain_contrastive(
    #     encoder, projection_head, pretrain_loader,
    #     epochs=settings.pretrain_epochs,
    #     device=device,
    #     lr=settings.learning_rate
    # )
    
    print("Model pretraining completed")
    return "success"


def train_classifier(**context):
    """Train the classifier model."""
    import sys
    sys.path.insert(0, "/opt/airflow")
    
    import torch
    from pathlib import Path
    
    from config.settings import settings
    from src.models import Encoder, TabTransformerClassifier
    from src.models.training import train_classifier, evaluate_model
    from src.utils import save_model
    from src.data import DatabaseManager
    
    print("Starting classifier training...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # TODO: Load pretrained encoder and actual training data
    # For now, this is a placeholder
    
    cat_max_dict = {i: 10 for i in range(10)}  # Placeholder
    
    encoder = Encoder(
        cnt_cat_features=10,
        cnt_num_features=10,
        cat_max_dict=cat_max_dict,
        d_model=settings.d_model,
        nhead=settings.nhead,
        num_layers=settings.num_layers,
        dim_feedforward=settings.dim_feedforward,
        dropout_rate=settings.dropout_rate,
    )
    
    model = TabTransformerClassifier(
        encoder=encoder,
        d_model=settings.d_model,
        final_hidden=settings.final_hidden,
        dropout_rate=settings.dropout_rate,
    )
    
    # TODO: Replace with actual data loaders
    # model, metrics = train_classifier(
    #     model, train_loader, val_loader,
    #     epochs=settings.num_epochs,
    #     device=device,
    #     lr=settings.learning_rate,
    #     patience=settings.early_stopping_patience
    # )
    
    # Generate model version
    model_version = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Save model
    model_config = {
        "d_model": settings.d_model,
        "nhead": settings.nhead,
        "num_layers": settings.num_layers,
        "dim_feedforward": settings.dim_feedforward,
        "dropout_rate": settings.dropout_rate,
        "final_hidden": settings.final_hidden,
    }
    
    # TODO: Uncomment when actual training is implemented
    # model_path_str = save_model(
    #     model,
    #     settings.model_path,
    #     model_version,
    #     {"model_config": model_config}
    # )
    
    # # Save to database
    # db = DatabaseManager(settings.database_url)
    # training_sample_count = context["ti"].xcom_pull(
    #     task_ids="prepare_training_data",
    #     key="training_sample_count"
    # )
    # db.save_model_metadata(
    #     model_version=model_version,
    #     model_path=model_path_str,
    #     training_samples=training_sample_count,
    #     metrics=metrics,
    #     training_config=model_config,
    #     is_active=False
    # )
    
    # Store model version in XCom
    context["ti"].xcom_push(key="model_version", value=model_version)
    
    print(f"Classifier training completed: {model_version}")
    return "success"


prepare_data_task = PythonOperator(
    task_id="prepare_training_data",
    python_callable=prepare_training_data,
    provide_context=True,
    dag=dag,
)

pretrain_task = PythonOperator(
    task_id="pretrain_model",
    python_callable=pretrain_model,
    provide_context=True,
    dag=dag,
)

train_task = PythonOperator(
    task_id="train_classifier",
    python_callable=train_classifier,
    provide_context=True,
    dag=dag,
)

prepare_data_task >> pretrain_task >> train_task
