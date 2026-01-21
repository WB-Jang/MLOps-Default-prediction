import sys
sys.path.insert(0, '/opt/airflow')
from src.database.mongodb import mongodb_client
from datetime import datetime
import torch
import os

print("=" * 60)
print("   모델 MongoDB 등록")
print("=" * 60)

model_path = '/opt/airflow/src/models/finetuned_model_f.pth'

print(f"\n[1] 모델 파일 확인")
if os.path.exists(model_path):
    file_size = os.path.getsize(model_path) / 1024
    print(f"✅ 파일 발견: {model_path}")
    print(f"   크기: {file_size:.2f} KB")
else:
    print(f"❌ 파일 없음: {model_path}")
    sys.exit(1)

print(f"\n[2] 모델 로드")
try:
    checkpoint = torch.load(model_path, map_location='cpu')
    print(f"✅ 모델 로드 성공")
    print(f"   Keys: {list(checkpoint.keys())}")
    metadata = checkpoint. get('metadata', {})
    if metadata:
        print(f"   Metadata 발견")
except Exception as e:
    print(f"⚠️ 로드 에러: {e}")
    metadata = {}

print(f"\n[3] MongoDB 등록")
mongodb_client.connect()

try:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    model_id = mongodb_client.store_model_metadata(
        model_name='default_prediction_classifier',
        model_path=model_path,
        model_version=f'finetuned_{timestamp}',
        hyperparameters={
            'source': 'finetuned_model',
            'original_file': 'finetuned_model_f. pth',
            'imported_at': timestamp,
            **metadata
        },
        metrics={
            'status': 'imported_finetuned',
            'file_size_kb': round(file_size, 2)
        }
    )
    
    print(f"✅ MongoDB 등록 완료!")
    print(f"   Model ID: {model_id}")
    
    print(f"\n[4] 등록 확인")
    model_doc = mongodb_client.get_model_metadata('default_prediction_classifier')
    if model_doc: 
        print(f"   - Name: {model_doc.get('model_name')}")
        print(f"   - Version: {model_doc.get('model_version')}")
        print(f"   - Path: {model_doc.get('model_path')}")
        print(f"   - Created: {model_doc. get('created_at')}")
    
finally:
    mongodb_client.disconnect()

print("\n" + "=" * 60)
print("   완료!")
print("=" * 60)
