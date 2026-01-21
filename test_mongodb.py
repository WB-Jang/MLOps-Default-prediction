import sys
sys.path.insert(0, '/opt/airflow')
from src.database.mongodb import mongodb_client

print("=" * 60)
print("   mongodb_client 테스트")
print("=" * 60)

try:
    print("\n[1] 연결 시도")
    mongodb_client. connect()
    print("✅ 연결 성공!")
    
    print("\n[2] 컬렉션 조회")
    collection = mongodb_client.get_collection("model_metadata")
    count = collection.count_documents({})
    print(f"   model_metadata 개수: {count}")
    
    if count > 0:
        print("\n[3] 최신 모델")
        latest = collection.find_one(sort=[("created_at", -1)])
        print(f"   - Name: {latest.get('model_name')}")
        print(f"   - Version: {latest.get('model_version')}")
        print(f"   - Path: {latest.get('model_path')}")
    
    mongodb_client.disconnect()
    print("\n✅ 테스트 완료!")
    
except Exception as e:
    print(f"\n❌ 에러:  {e}")
    import traceback
    traceback. print_exc()

print("=" * 60)
