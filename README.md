# Loan Default Prediction - MLOps Package

은행 여신 계좌의 Default를 예측하기 위한 완전한 MLOps 패키지입니다. 데이터 수집부터 모델 검증, 자동 재학습 및 배포까지 전체 파이프라인을 포함합니다.

## 주요 기능

- 🤖 **Transformer 기반 딥러닝 모델**: TabTransformer 아키텍처를 사용한 loan default 예측
- 🔄 **자동화된 MLOps 파이프라인**: Airflow를 통한 전체 워크플로우 자동화
- 📊 **성능 모니터링**: F1-score 기반 모델 성능 자동 모니터링 (threshold: 0.8)
- 🔁 **자동 재학습**: 성능이 임계값 이하로 떨어지면 자동으로 재학습 트리거
- 🐳 **완전한 컨테이너화**: Docker 및 DevContainer를 통한 일관된 개발/배포 환경
- 💾 **PostgreSQL 데이터베이스**: 데이터, 모델 메타데이터, 성능 메트릭 저장
- 🚀 **FastAPI 서빙**: RESTful API를 통한 실시간 예측 서비스
- 📦 **Poetry 의존성 관리**: 재현 가능한 Python 환경

## 아키텍처

```
┌─────────────────┐
│  Data Sources   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────┐
│   PostgreSQL    │────▶│   Airflow    │
│   (Data Store)  │     │  (Workflow)  │
└─────────────────┘     └──────┬───────┘
                               │
                    ┌──────────┼──────────┐
                    ▼          ▼          ▼
              ┌─────────┐ ┌────────┐ ┌──────────┐
              │  Data   │ │ Model  │ │  Model   │
              │Collection│ │Training│ │Validation│
              └─────────┘ └────────┘ └──────────┘
                                           │
                                           ▼
                                    ┌──────────────┐
                                    │ Auto Retrain │
                                    │ (F1 < 0.8)   │
                                    └──────┬───────┘
                                           │
                                           ▼
                                    ┌──────────────┐
                                    │   FastAPI    │
                                    │  (Serving)   │
                                    └──────────────┘
```

## 프로젝트 구조

```
.
├── .devcontainer/          # VSCode DevContainer 설정
├── airflow/
│   └── dags/              # Airflow DAG 정의
│       ├── data_collection_dag.py
│       ├── model_training_dag.py
│       └── model_validation_deployment_dag.py
├── config/                # 설정 파일
│   ├── init.sql          # PostgreSQL 초기화 스크립트
│   └── settings.py       # 애플리케이션 설정
├── src/
│   ├── api/              # FastAPI 애플리케이션
│   │   └── main.py
│   ├── data/             # 데이터 관리
│   │   └── database.py
│   ├── models/           # 모델 아키텍처
│   │   ├── transformer.py
│   │   ├── augmentation.py
│   │   └── training.py
│   └── utils/            # 유틸리티 함수
│       └── model_utils.py
├── tests/                # 테스트 코드
├── docker-compose.yml    # Docker Compose 설정
├── Dockerfile           # Docker 이미지 정의
├── pyproject.toml       # Poetry 의존성 관리
└── README.md            # 이 파일
```

## 시작하기

### 사전 요구사항

- Docker & Docker Compose
- Python 3.9+
- Poetry (선택사항, 로컬 개발용)

### 설치 및 실행

1. **저장소 클론**

```bash
git clone https://github.com/WB-Jang/MLOps-Default-prediction.git
cd MLOps-Default-prediction
```

2. **환경 변수 설정**

```bash
cp .env.example .env
# .env 파일을 필요에 맞게 수정
```

3. **Docker Compose로 전체 스택 실행**

```bash
docker-compose up -d
```

이 명령어는 다음 서비스를 시작합니다:
- PostgreSQL (포트 5432)
- Airflow Webserver (포트 8080)
- Airflow Scheduler
- FastAPI Application (포트 8000)

4. **서비스 확인**

- Airflow UI: http://localhost:8080
- FastAPI Docs: http://localhost:8000/docs
- PostgreSQL: localhost:5432

### DevContainer를 사용한 개발

VSCode에서 DevContainer를 사용하면 일관된 개발 환경을 제공받을 수 있습니다:

1. VSCode에서 프로젝트 열기
2. "Reopen in Container" 선택
3. 컨테이너 내부에서 개발 시작

## 사용 방법

### 1. 데이터 수집

Airflow UI에서 `data_collection` DAG를 활성화합니다:

```python
# 데이터는 PostgreSQL의 loan_data.raw_data 테이블에 저장됩니다
```

### 2. 모델 학습

`model_training` DAG를 트리거하여 새 모델을 학습합니다:

```bash
# Airflow UI에서 "model_training" DAG 트리거
# 또는 CLI 사용
docker exec loan_airflow_scheduler airflow dags trigger model_training
```

학습 프로세스:
1. 데이터 준비
2. Contrastive Learning을 통한 사전학습
3. 분류 모델 학습
4. 모델 저장 및 메타데이터 기록

### 3. 모델 검증 및 배포

`model_validation_deployment` DAG가 매일 자동으로 실행되어:

1. 활성 모델의 성능 평가
2. F1-score가 0.8 미만인지 확인
3. 성능이 낮으면 자동으로 재학습 트리거
4. 새로운 모델 배포

### 4. 예측 API 사용

```python
import requests

# 예측 요청
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "categorical_features": {
            "feature1": 1,
            "feature2": 2,
            # ...
        },
        "numerical_features": {
            "feature3": 0.5,
            "feature4": 1.2,
            # ...
        }
    }
)

result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Probability: {result['probability']}")
```

## 모델 아키텍처

### TabTransformer

Transformer 기반 아키텍처를 사용하여 tabular data를 처리합니다:

- **Encoder**: 범주형 및 수치형 feature를 임베딩하고 Transformer로 처리
- **Classifier**: 사전학습된 encoder에 분류 헤드 추가
- **Contrastive Learning**: NTXent loss를 사용한 사전학습

### 주요 하이퍼파라미터

```python
D_MODEL = 32              # 임베딩 차원
NHEAD = 4                 # Attention head 수
NUM_LAYERS = 6            # Transformer layer 수
DIM_FEEDFORWARD = 64      # Feedforward 차원
DROPOUT_RATE = 0.3        # Dropout 비율
LEARNING_RATE = 0.0003    # 학습률
BATCH_SIZE = 32           # 배치 크기
```

## 데이터베이스 스키마

### 주요 테이블

- `loan_data.raw_data`: 원시 데이터
- `loan_data.processed_data`: 전처리된 데이터
- `loan_data.model_metadata`: 모델 메타데이터
- `loan_data.predictions`: 예측 결과
- `loan_data.model_performance`: 모델 성능 기록

## 모니터링 및 재학습

### 자동 재학습 트리거 조건

1. **F1-score 임계값**: 최근 7일간 평균 F1-score < 0.8
2. **새로운 데이터**: 지정된 수 이상의 새로운 데이터 축적

### 성능 메트릭

- F1-score
- Accuracy
- Precision
- Recall
- ROC-AUC

## 개발

### 로컬 개발 환경 설정

```bash
# Poetry 설치
pip install poetry

# 의존성 설치
poetry install

# 가상환경 활성화
poetry shell

# 개발 서버 실행
uvicorn src.api.main:app --reload
```

### 테스트 실행

```bash
poetry run pytest tests/
```

### 코드 포맷팅

```bash
# Black 포맷터
poetry run black src/

# Import 정렬
poetry run isort src/

# Linting
poetry run flake8 src/
```

## 설정

주요 설정은 `.env` 파일 또는 환경 변수로 관리됩니다:

```bash
DATABASE_URL=postgresql://mlops_user:mlops_password@localhost:5432/loan_default
F1_SCORE_THRESHOLD=0.8
RETRAINING_SAMPLE_THRESHOLD=1000
MODEL_PATH=./models
```

## 로깅

- Airflow 로그: `airflow/logs/`
- 애플리케이션 로그: 콘솔 출력

## 트러블슈팅

### PostgreSQL 연결 오류

```bash
# PostgreSQL 컨테이너 상태 확인
docker-compose ps postgres

# 로그 확인
docker-compose logs postgres
```

### Airflow DAG가 보이지 않음

```bash
# DAG 폴더 권한 확인
ls -la airflow/dags/

# Airflow 스케줄러 재시작
docker-compose restart airflow-scheduler
```

### 모델 로딩 오류

```bash
# 모델 파일 존재 확인
ls -la models/

# 데이터베이스의 활성 모델 확인
docker exec -it loan_postgres psql -U mlops_user -d loan_default -c "SELECT * FROM loan_data.model_metadata WHERE is_active = true;"
```

## 기여

프로젝트에 기여하고 싶으신 분은 Pull Request를 보내주세요.

## 라이센스

이 프로젝트는 MIT 라이센스를 따릅니다.

## 참고 자료

- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Poetry Documentation](https://python-poetry.org/docs/)

## 연락처

문의사항이 있으시면 이슈를 생성해주세요.
