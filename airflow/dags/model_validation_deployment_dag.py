"""Airflow DAG for model validation and deployment."""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

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
    "model_validation_deployment",
    default_args=default_args,
    description="Validate and deploy model with F1-score threshold check",
    schedule_interval="@daily",
    catchup=False,
    tags=["model", "validation", "deployment"],
)


def evaluate_active_model(**context):
    """Evaluate the currently active model."""
    import sys
    sys.path.insert(0, "/opt/airflow")
    
    from config.settings import settings
    from src.data import DatabaseManager
    
    db = DatabaseManager(settings.database_url)
    
    # Get active model
    active_model = db.get_active_model()
    
    if not active_model:
        print("No active model found")
        context["ti"].xcom_push(key="needs_retraining", value=True)
        context["ti"].xcom_push(key="f1_score", value=0.0)
        return "needs_retraining"
    
    print(f"Evaluating model: {active_model['model_version']}")
    
    # TODO: Load model and evaluate on recent data
    # For now, use stored F1 score
    f1_score = active_model.get("f1_score", 0.0)
    
    # Check if retraining is needed based on recent performance
    needs_retraining = db.check_retraining_needed(settings.f1_score_threshold)
    
    context["ti"].xcom_push(key="needs_retraining", value=needs_retraining)
    context["ti"].xcom_push(key="f1_score", value=f1_score)
    context["ti"].xcom_push(key="model_version", value=active_model["model_version"])
    
    print(f"Model F1 Score: {f1_score}")
    print(f"Needs Retraining: {needs_retraining}")
    
    return "success"


def check_performance_threshold(**context):
    """Check if model performance is below threshold."""
    import sys
    sys.path.insert(0, "/opt/airflow")
    
    from config.settings import settings
    
    needs_retraining = context["ti"].xcom_pull(
        task_ids="evaluate_active_model",
        key="needs_retraining"
    )
    
    f1_score = context["ti"].xcom_pull(
        task_ids="evaluate_active_model",
        key="f1_score"
    )
    
    print(f"Current F1 Score: {f1_score}")
    print(f"Threshold: {settings.f1_score_threshold}")
    
    if needs_retraining or f1_score < settings.f1_score_threshold:
        print("Performance below threshold - triggering retraining")
        return "trigger_retraining"
    else:
        print("Performance acceptable - no retraining needed")
        return "skip_retraining"


def trigger_retraining(**context):
    """Prepare for model retraining."""
    print("Preparing to trigger model retraining...")
    return "success"


def skip_retraining(**context):
    """Skip retraining when performance is acceptable."""
    print("Model performance is acceptable. Skipping retraining.")
    return "success"


def deploy_new_model(**context):
    """Deploy newly trained model if available."""
    import sys
    sys.path.insert(0, "/opt/airflow")
    
    from config.settings import settings
    from src.data import DatabaseManager
    
    db = DatabaseManager(settings.database_url)
    
    # Check if there's a new model to deploy
    # (This would be triggered after successful training)
    
    # TODO: Implement actual deployment logic
    # - Load new model
    # - Validate performance
    # - If F1 > threshold, set as active
    # - Update database
    
    print("Checking for new models to deploy...")
    
    # Example deployment logic:
    # if new_model_f1 >= settings.f1_score_threshold:
    #     db.set_active_model(new_model_version)
    #     print(f"Deployed new model: {new_model_version}")
    # else:
    #     print(f"New model F1 ({new_model_f1}) below threshold ({settings.f1_score_threshold})")
    
    return "success"


def save_performance_metrics(**context):
    """Save current performance metrics to database."""
    import sys
    sys.path.insert(0, "/opt/airflow")
    
    from datetime import date
    from config.settings import settings
    from src.data import DatabaseManager
    
    db = DatabaseManager(settings.database_url)
    
    model_version = context["ti"].xcom_pull(
        task_ids="evaluate_active_model",
        key="model_version"
    )
    
    f1_score = context["ti"].xcom_pull(
        task_ids="evaluate_active_model",
        key="f1_score"
    )
    
    if model_version:
        # TODO: Calculate full metrics on evaluation set
        metrics = {
            "f1_score": f1_score,
            "accuracy": 0.0,  # Placeholder
            "precision": 0.0,  # Placeholder
            "recall": 0.0,  # Placeholder
            "roc_auc": 0.0,  # Placeholder
        }
        
        db.save_model_performance(
            model_version=model_version,
            evaluation_date=date.today(),
            metrics=metrics,
            sample_count=0,  # Placeholder
        )
        
        print(f"Saved performance metrics for {model_version}")
    
    return "success"


evaluate_task = PythonOperator(
    task_id="evaluate_active_model",
    python_callable=evaluate_active_model,
    provide_context=True,
    dag=dag,
)

check_threshold_task = BranchPythonOperator(
    task_id="check_performance_threshold",
    python_callable=check_performance_threshold,
    provide_context=True,
    dag=dag,
)

trigger_retraining_task = TriggerDagRunOperator(
    task_id="trigger_retraining",
    trigger_dag_id="model_training",
    wait_for_completion=False,
    dag=dag,
)

skip_retraining_task = PythonOperator(
    task_id="skip_retraining",
    python_callable=skip_retraining,
    provide_context=True,
    dag=dag,
)

save_metrics_task = PythonOperator(
    task_id="save_performance_metrics",
    python_callable=save_performance_metrics,
    provide_context=True,
    dag=dag,
)

deploy_task = PythonOperator(
    task_id="deploy_new_model",
    python_callable=deploy_new_model,
    provide_context=True,
    dag=dag,
)

evaluate_task >> check_threshold_task
check_threshold_task >> trigger_retraining_task
check_threshold_task >> skip_retraining_task
skip_retraining_task >> save_metrics_task
trigger_retraining_task >> deploy_task
