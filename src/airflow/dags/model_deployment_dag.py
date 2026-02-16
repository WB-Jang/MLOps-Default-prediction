"""Airflow DAG for model deployment pipeline."""
from datetime import datetime, timedelta
from airflow.utils import timezone
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.python import PythonSensor
# from airflow.sensors.external_task import ExternalTaskSensor
from airflow.utils.dates import days_ago
import torch
import os
import shutil

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.database.mongodb import mongodb_client
from config.settings import settings
from loguru import logger


default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


def check_for_approved_model(**context):
    """MongoDB에서 배포 대기 중인(approved/retrained) 모델이 있는지 확인."""
    logger.info("Polling MongoDB for approved model...")
    
    mongodb_client.connect()
    try:
        collection = mongodb_client.get_collection("model_metadata")
        # 'deployed' 상태가 아니면서 'approved' 또는 'retrained'인 최신 모델 검색
        latest_model = collection.find_one(
            {
                "model_name": "default_prediction_classifier", 
                "status": {"$in": ["approved", "retrained"]}
            },
            sort=[("created_at", -1)]
        )
        
        if latest_model:
            logger.info(f"✅ Found model ready for deployment: {latest_model['model_version']}")
            # 찾은 모델 정보를 XCom에 저장 (다음 태스크에서 사용)
            context['ti'].xcom_push(key='target_model', value={
                "model_id": str(latest_model['_id']),
                "model_path": latest_model['model_path'],
                "model_version": latest_model['model_version']
            })
            return True # True를 리턴하면 센서가 완료됨
        
        logger.info("⏳ No approved model found yet. Waiting...")
        return False # False면 poke_interval 후에 다시 실행
        
    finally:
        mongodb_client.disconnect()

def deploy_model(**context):
    """센서가 찾아낸 모델 정보를 가져와 배포 수행."""
    ti = context['ti']
    # 1. 센서에서 전달한 모델 메타데이터 가져오기
    model_info = ti.xcom_pull(task_ids='sensor_wait_for_approved_model', key='target_model')
    
    if not model_info:
        raise ValueError("No model info found in XCom. Something went wrong with the sensor.")

    # 2. 필요한 변수 정의 (에러 방지)
    model_version = model_info['model_version']
    model_path = model_info['model_path']
    
    # 배포 경로 설정 (settings.model_save_path 사용)
    deployment_dir = os.path.join(settings.model_save_path, "deployed")
    os.makedirs(deployment_dir, exist_ok=True)
    deployed_model_path = os.path.join(deployment_dir, "current_model.pth")

    logger.info(f"🚀 Deploying model version: {model_version}")

    try:
        # 3. 실제 배포 작업 (파일 복사)
        if os.path.exists(model_path):
            shutil.copy2(model_path, deployed_model_path)
            logger.info(f"✅ Model copied to: {deployed_model_path}")
        else:
            raise FileNotFoundError(f"Source model file not found at {model_path}")

        # 4. 다음 태스크를 위한 데이터 반환 (XCom 저장)
        return {
            "model_id": model_info['model_id'],
            "model_version": model_version,
            "deployment_path": deployed_model_path,
            "deployment_timestamp": timezone.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"❌ Deployment failed: {e}")
        raise # 에러를 다시 발생시켜 Airflow가 실패로 인식하게 함

def notify_deployment_complete(**context):
    logger.info("Deployment pipeline completed")
    ti = context['ti']
    
    # 1. include_prior_dates는 제거하고 현재 DagRun의 데이터만 신뢰합니다.
    deployment_info = ti.xcom_pull(task_ids='deploy_model')
    
    # 2. [변경] 데이터가 없으면 진행하지 않고 에러를 내서 원인을 파악하게 합니다.
    if deployment_info is None:
        raise ValueError("❌ 'deploy_model'로부터 데이터를 전달받지 못했습니다. 업스트림 로그를 확인하세요.")

    model_ver = deployment_info['model_version']
    logger.info(f"Model {model_ver} successfully deployed")
    
    # Store notification
    mongodb_client.connect()
    try:
        notification_collection = mongodb_client.get_collection("notifications")
        notification_collection.insert_one({
            "type": "deployment_complete",
            "timestamp": timezone.utcnow(), # Airflow 권장 방식
            "model_id": deployment_info['model_id'],
            "model_version": deployment_info['model_version'],
            "message": f"Model {deployment_info['model_version']} deployed successfully"
        })
        
        # TODO: Send actual notifications (email, Slack, etc.)
        # Example: send_slack_notification(deployment_info)
        
    finally:
        mongodb_client.disconnect()


# Define DAG
with DAG(
    'model_deployment_pipeline',
    default_args=default_args,
    schedule_interval=timedelta(days=1),
    catchup=False,
    max_active_runs=1,
    tags=['deployment', 'mongodb'],
) as dag:

    # [변경됨] 외부 태스크 센서 대신 MongoDB 직접 감시 센서
    wait_for_model_sensor = PythonSensor(
        task_id='sensor_wait_for_approved_model',
        python_callable=check_for_approved_model,
        mode='reschedule',    # 워커 슬롯 반납 모드 유지
        poke_interval=60,     # 1분마다 DB 확인
        timeout=3600,         # 1시간 동안 안 나오면 실패 처리
    )

    deploy_task = PythonOperator(
        task_id='deploy_model',
        python_callable=deploy_model,
    )

    notify_task = PythonOperator(
        task_id='notify_deployment_complete',
        python_callable=notify_deployment_complete,
    )

    wait_for_model_sensor >> deploy_task >> notify_task