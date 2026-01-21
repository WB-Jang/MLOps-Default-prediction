import sys
sys.path. insert(0, '/opt/airflow')
from src.database.mongodb import mongodb_client
from datetime import datetime
import torch
import os

print("=" * 60)
print("   모델 MongoDB 등록")
print("=" * 60)

model_path = '/opt/airflow/src/models/finetuned_model_f. pth'

print(f"\n[1] 모델 로드")
file_size = os.path.getsize(model_path) / 1024
checkpoint = torch.load(model_path, map_location='cpu')
print(f"✅ 크기: {file_size:.2f} KB")

metadata = checkpoint.get('metadata', {})

print(f"\n[2] MongoDB 연결")
mongodb_client.connect()
print("✅ 연결 성공")

try:
    print(f"\n[3] 모델 메타데이터 저장")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    model_id = mongodb_client.store_model_metadata(
        model_name='default_prediction_classifier',
        model_path=model_path,
        model_version=f'finetuned_{timestamp}',
        hyperparameters={'source': 'finetuned', **metadata},
        metrics={'status': 'imported', 'file_size_kb':  round(file_size, 2)}
    )
    
    print(f"✅ 저장 완료!  Model ID: {model_id}")
    
    model_doc = mongodb_client.get_model_metadata('default_prediction_classifier')
    if model_doc:
        print(f"\n[4] 등록 확인")
        print(f"   - Name: {model_doc.get('model_name')}")
        print(f"   - Version: {model_doc.get('model_version')}")
        print(f"   - Path: {model_doc. get('model_path')}")
    
finally:
    mongodb_client.disconnect()

print("\n" + "=" * 60)
