# MLOps 파이프라인 워크플로우 문서

## 개요

본 레파지토리는 일별로 반복되는 MLOps 파이프라인을 구현합니다. 이 문서는 각 단계의 흐름과 구현 세부사항을 설명합니다.

## 전체 워크플로우

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1. 초기 모델 저장 (MongoDB)                                         │
│    - Pre-trained Encoder Model                                      │
│    - Fine-tuned Classifier Model                                    │
└─────────────────┬───────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 2. 일별 데이터 생성 (data_pipeline_dag)                            │
│    - Distribution model 활용 Synthetic data 생성                   │
│    - MongoDB raw_data에 저장                                        │
└─────────────────┬───────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 3. 데이터 전처리 (data_pipeline_dag)                               │
│    - Raw data 로딩 및 전처리                                        │
│    - MongoDB processed_data에 저장                                  │
└─────────────────┬───────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 4. 모델 평가 (model_evaluation_pipeline)                           │
│    - 최신 fine-tuned 모델 로딩                                      │
│    - Test data로 평가 (corp_default_modeling_f.py 코드 사용)       │
│    - F1 score 체크                                                  │
└─────────────────┬───────────────────────────────────────────────────┘
                  │
                  ▼
         ┌────────┴────────┐
         │                 │
    F1 >= 0.75        F1 < 0.75
         │                 │
         │                 ▼
         │    ┌─────────────────────────────────────────────┐
         │    │ 5. 재학습 시작 (최대 3회 시도)              │
         │    │    - 평가 데이터로 fine-tuning 수행         │
         │    │    - corp_default_modeling_f.py 코드 사용   │
         │    └─────────────┬───────────────────────────────┘
         │                  │
         │                  ▼
         │         ┌────────┴────────┐
         │         │                 │
         │    F1 >= 0.75        F1 < 0.75
         │         │                 │
         │         │          ┌──────┴──────┐
         │         │          │             │
         │         │     시도 < 3회     시도 >= 3회
         │         │          │             │
         │         │          │             ▼
         │         │          │    ┌─────────────────┐
         │         │          │    │ 6. Alert 발송   │
         │         │          │    │ - MongoDB 저장  │
         │         │          │    └─────────────────┘
         │         │          │
         │         │          └─── 4번으로 돌아가기 (재시도)
         │         │
         └─────────┴──────────────────────────┐
                                              │
                                              ▼
                            ┌─────────────────────────────────┐
                            │ 7. 모델 저장 (MongoDB)          │
                            │    - 새 버전으로 저장           │
                            │    - 평가 결과 저장             │
                            └─────────────┬───────────────────┘
                                          │
                                          ▼
                            ┌─────────────────────────────────┐
                            │ 8. 모델 배포                    │
                            │ (model_deployment_pipeline)     │
                            │    - 승인된 모델 배포           │
                            │    - 배포 경로로 복사           │
                            └─────────────────────────────────┘
```

## 세부 구현

### 1. 초기 모델 준비

**위치**: `src/models/pretrained_model_f.pth`, `src/models/finetuned_model_f.pth`

**설명**: 
- Pre-training과 fine-tuning이 이미 완료된 모델을 MongoDB에 저장
- `corp_default_modeling_f.py`의 코드로 학습된 모델

**저장 위치**: MongoDB `model_metadata` collection

### 2. 일별 데이터 생성

**DAG**: `data_pipeline_dag` (data_ingestion_dag.py)

**Task**: `generate_data`

**코드 위치**: `src/airflow/dags/data_ingestion_dag.py`

**구현**:
```python
def generate_data_task(**context):
    """Distribution model을 사용하여 합성 데이터 생성"""
    processor = DataGenLoaderProcessor(data_path=DATA_PATH)
    processor.data_generation(data_quantity=10000)
```

**특징**:
- `src/data/distribution_model.pkl` 활용
- SDV의 GaussianCopulaSynthesizer 사용
- 실제 데이터 분포를 학습한 모델로 realistic synthetic data 생성
- Fallback: distribution model 없으면 대체 방법 사용

**저장**: 
- CSV: `./data/raw/synthetic_data.csv`
- MongoDB: `raw_data` collection (GridFS)

### 3. 데이터 전처리 및 저장

**DAG**: `data_pipeline_dag`

**Task**: `preprocess_data`

**구현**:
```python
def preprocess_data_task(**context):
    """데이터 전처리 및 MongoDB 저장"""
    processor = DataGenLoaderProcessor(data_path=DATA_PATH)
    
    # 전처리 파이프라인
    df = processor.load_raw_data()
    df = processor.IsNull_cols()
    df = processor.obj_cols()
    processor.dt_data_handling()
    processor.data_shrinkage()
    scaled_df, str_cols, num_cols, nunique = processor.detecting_type_encoding()
    
    # MongoDB 저장
    processed_data_records = scaled_df.to_dict('records')
    collection.insert_many(processed_data_records)
```

**저장**:
- CSV: `./data/processed/processed_data.csv`
- MongoDB: `processed_data` collection
- Metadata: `processing_metadata` collection

### 4. 모델 평가

**DAG**: `model_evaluation_pipeline`

**Tasks**: 
- `load_latest_model`: MongoDB에서 최신 모델 로딩
- `evaluate_model`: 테스트 데이터로 평가

**코드 위치**: 
- `src/airflow/dags/model_evaluation_dag.py`
- `src/models/evaluation.py`

**구현**:
```python
# corp_default_modeling_f.py에서 추출한 evaluate_model 함수 사용
from src.models.evaluation import evaluate_model as eval_model_func

def evaluate_model(**context):
    # 모델 로딩
    classifier = TabTransformerClassifier(...)
    checkpoint = torch.load(model_path)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    
    # corp_default_modeling_f.py의 평가 코드 사용
    metrics = eval_model_func(classifier, test_loader, device=device)
    # metrics: {"accuracy", "roc_auc", "f1_score"}
```

**F1 Threshold**: 0.75 (`F1_THRESHOLD = 0.75`)

### 5. 조건부 재학습 (최대 3회)

**DAG**: `model_evaluation_pipeline`

**Tasks**:
- `check_model_performance`: F1 score 확인 및 분기
- `prepare_retraining_data`: 재학습 데이터 준비
- `finetune_model`: Fine-tuning 수행
- `evaluate_finetuned_model`: 재학습 모델 평가

**재시도 메커니즘**:
```python
MAX_RETRAIN_ATTEMPTS = 3

def check_model_performance(**context):
    retry_count = ti.xcom_pull(key='retry_count', default=0)
    
    if metrics['f1_score'] < F1_THRESHOLD:
        if retry_count < MAX_RETRAIN_ATTEMPTS:
            # 재학습 시도
            ti.xcom_push(key='retry_count', value=retry_count + 1)
            return 'prepare_retraining_data'
        else:
            # 최대 시도 초과 - Alert
            return 'send_alert'
    else:
        # 성공
        return 'send_evaluation_results'
```

**Fine-tuning 구현** (corp_default_modeling_f.py 코드 사용):
```python
from src.models.evaluation import finetune_model as finetune_func

def finetune_model(**context):
    # 클래스 가중치 계산 (불균형 데이터 처리)
    class_weights = compute_class_weight(
        class_weight="balanced", 
        classes=classes, 
        y=y_fine
    )
    
    # corp_default_modeling_f.py의 fine-tuning 코드 사용
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    training_metrics, test_metrics = finetune_func(
        model=classifier,
        train_loader=fine_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=3
    )
```

**Loop-back 로직**:
```python
def evaluate_finetuned_model(**context):
    """재학습 모델 평가 후 다음 액션 결정"""
    test_metrics = finetuned_info['test_metrics']
    retry_count = ti.xcom_pull(key='retry_count')
    
    if test_metrics['f1_score'] < F1_THRESHOLD:
        if retry_count < MAX_RETRAIN_ATTEMPTS:
            return 'check_model_performance'  # Loop back
        else:
            return 'send_alert'
    else:
        return 'send_evaluation_results'
```

### 6. Alert 발송 (3회 실패 시)

**Task**: `send_alert`

**구현**:
```python
def send_alert(**context):
    """최대 재시도 후 실패 시 Alert 발송"""
    alert_message = {
        "alert_type": "MODEL_PERFORMANCE_FAILURE",
        "severity": "CRITICAL",
        "f1_score": final_f1,
        "f1_threshold": F1_THRESHOLD,
        "retry_attempts": retry_count,
        "message": f"Model failed to meet F1 threshold after {retry_count} attempts"
    }
    
    # MongoDB 저장
    alert_collection.insert_one(alert_message)
    
    # TODO: 실제 알림 시스템 연동
    # - Email
    # - Slack
    # - PagerDuty
```

**저장 위치**: 
- MongoDB `alerts` collection
- MongoDB `evaluation_events` collection (tracking)

### 7. 모델 버전 관리 및 저장

**구현**:
```python
def finetune_model(**context):
    # 새 버전으로 모델 저장
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    finetuned_path = f"classifier_finetuned_{timestamp}.pth"
    
    torch.save({
        'model_state_dict': classifier.state_dict(),
        'metadata': metadata,
        'finetuned_from': model_info['model_id'],
        'finetune_timestamp': timestamp
    }, finetuned_path)
    
    # MongoDB에 메타데이터 저장
    mongodb_client.store_model_metadata(
        model_name="default_prediction_classifier",
        model_path=finetuned_path,
        model_version=timestamp,
        hyperparameters={...},
        metrics=test_metrics
    )
```

**저장 위치**:
- 모델 파일: `./models/classifier_finetuned_YYYYMMDD_HHMMSS.pth`
- MongoDB: `model_metadata` collection

### 8. 모델 배포

**DAG**: `model_deployment_pipeline` (model_deployment_dag.py)

**Tasks**:
- `wait_for_evaluation`: 평가 완료 대기 (ExternalTaskSensor)
- `wait_for_model_ready`: 승인된 모델 확인
- `deploy_model`: 모델 배포
- `notify_deployment_complete`: 배포 완료 알림

**구현**:
```python
def deploy_model(**context):
    """승인된 모델을 프로덕션에 배포"""
    deployment_dir = os.path.join(settings.model_save_path, "deployed")
    deployed_model_path = os.path.join(deployment_dir, "current_model.pth")
    
    # 모델 복사
    shutil.copy2(model_path, deployed_model_path)
    
    # 버전 정보 저장
    with open(version_file, 'w') as f:
        f.write(f"Version: {model_version}\n")
        f.write(f"Deployed at: {datetime.utcnow().isoformat()}\n")
    
    # MongoDB에 배포 이벤트 저장
    deployment_collection.insert_one({
        "event_type": "model_deployed",
        "model_id": model_id,
        "deployment_path": deployed_model_path,
        "status": "deployed"
    })
```

**저장 위치**:
- 배포 경로: `./models/deployed/current_model.pth`
- MongoDB: `deployment_events` collection

## DAG 스케줄링

모든 DAG는 일별(daily) 실행으로 스케줄링되어 있습니다:

```python
schedule_interval=timedelta(days=1)
```

### DAG 실행 순서

1. **data_pipeline_dag** (매일 실행)
   - 데이터 생성 → 저장 → 전처리 → 저장

2. **model_evaluation_pipeline** (매일 실행)
   - 모델 평가 → 필요시 재학습 (최대 3회) → Alert or 성공

3. **model_deployment_pipeline** (매일 실행)
   - 평가 완료 대기 → 승인된 모델 배포

### DAG 간 연동

```python
# model_deployment_pipeline에서 evaluation 완료 대기
wait_for_evaluation = ExternalTaskSensor(
    task_id='wait_for_evaluation',
    external_dag_id='model_evaluation_pipeline',
    external_task_id='send_evaluation_results',
    allowed_states=['success'],
    timeout=3600,
    poke_interval=300
)
```

## MongoDB Collections

### 데이터 관련
- **raw_data**: 원시 데이터 (GridFS)
- **processed_data**: 전처리된 데이터
- **processing_metadata**: 전처리 메타데이터

### 모델 관련
- **model_metadata**: 모델 메타데이터 (경로, 버전, 하이퍼파라미터)
- **performance_metrics**: 모델 평가 메트릭
- **training_events**: 학습 이벤트 로그
- **evaluation_events**: 평가 이벤트 로그
- **deployment_events**: 배포 이벤트 로그

### 알림 관련
- **alerts**: Alert 메시지
- **notifications**: 일반 알림

## 핵심 설정

### config/settings.py
```python
# Model paths
MODEL_SAVE_PATH = "./models"

# Model hyperparameters
D_MODEL = 32
NHEAD = 4
NUM_LAYERS = 6
DIM_FEEDFORWARD = 64
DROPOUT_RATE = 0.3

# Training
BATCH_SIZE = 32
LEARNING_RATE = 0.0003
EPOCHS = 10
```

### Evaluation settings
```python
# src/airflow/dags/model_evaluation_dag.py
F1_THRESHOLD = 0.75
MAX_RETRAIN_ATTEMPTS = 3
```

## 코드 재사용

`corp_default_modeling_f.py`의 핵심 평가/학습 로직이 다음과 같이 모듈화되었습니다:

### src/models/evaluation.py

```python
def evaluate_model(model, test_loader, device='cpu') -> Dict[str, float]:
    """
    corp_default_modeling_f.py에서 추출한 평가 함수
    
    Returns:
        {"accuracy", "roc_auc", "f1_score"}
    """
    # 원본 코드의 평가 로직 사용

def finetune_model(model, train_loader, test_loader, criterion, 
                   optimizer, device='cpu', epochs=3):
    """
    corp_default_modeling_f.py에서 추출한 fine-tuning 함수
    
    - 클래스 가중치 적용
    - 3 epochs fine-tuning
    - 훈련 및 테스트 메트릭 반환
    """
    # 원본 코드의 fine-tuning 로직 사용
```

## 실행 방법

### 로컬 실행 (개발)

```bash
# 1. 데이터 생성
python generate_raw_data.py --num-rows 10000

# 2. MongoDB 시작
docker run -d --name mongodb -p 27017:27017 mongo:7.0

# 3. 모델 학습 (초기 모델 생성)
python train.py --pretrain --train --epochs 10

# 4. MongoDB에 초기 모델 등록
python -c "from src.database.mongodb import mongodb_client; ..."
```

### Docker Compose 실행 (프로덕션)

```bash
# 1. 모든 서비스 시작
docker-compose up -d

# 2. Airflow UI 접속
# http://localhost:8080

# 3. DAG 활성화
# - data_pipeline_dag
# - model_evaluation_pipeline
# - model_deployment_pipeline

# 4. DAG 수동 트리거 또는 스케줄 대기
```

## 모니터링

### Airflow UI
- DAG 실행 상태 확인
- Task 로그 확인
- 재시도 상태 확인

### MongoDB 쿼리
```javascript
// Alert 확인
db.alerts.find().sort({timestamp: -1})

// 최신 평가 결과
db.evaluation_events.find().sort({timestamp: -1}).limit(10)

// 배포된 모델
db.deployment_events.find({status: "deployed"}).sort({timestamp: -1})

// 재학습 이력
db.model_metadata.find({finetuned_from: {$exists: true}})
```

## 문제 해결

### Q: F1 score가 계속 threshold 이하인 경우?
A: 
1. Alert가 발송되고 MongoDB에 기록됩니다
2. `alerts` collection 확인
3. 데이터 품질 또는 모델 아키텍처 검토 필요

### Q: 재학습이 무한 반복되는 경우?
A: 
- 최대 3회로 제한되어 있습니다 (`MAX_RETRAIN_ATTEMPTS`)
- 3회 후에는 자동으로 alert 발송

### Q: 배포가 실패하는 경우?
A:
1. `deployment_events` collection에서 에러 확인
2. 파일 권한 확인
3. 디스크 공간 확인

## 향후 개선 사항

1. **Alert 시스템 통합**
   - Email 알림
   - Slack 통합
   - PagerDuty 연동

2. **배포 전략**
   - Blue-Green 배포
   - Canary 배포
   - A/B 테스트

3. **모니터링 강화**
   - Prometheus 메트릭
   - Grafana 대시보드
   - 모델 drift 감지

4. **자동화**
   - 하이퍼파라미터 최적화
   - AutoML 통합
   - 데이터 검증 자동화

## 참고 문서

- README.md: 전체 프로젝트 설명
- ARCHITECTURE.md: 시스템 아키텍처
- docs/MODEL_EVALUATION_DAG_FLOW.md: 평가 DAG 상세 설명
- corp_default_modeling_f.py: 원본 모델 학습 코드
