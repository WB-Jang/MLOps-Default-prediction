import pandas as pd 
from sklearn.preprocessing import LabelEncoder
print('---Start---')
raw_data = pd.read_csv('C:/Users/1598505/OneDrive - Standard Chartered Bank/5.Python/jupyter_notebook/1.Data/MASTER1_v2_20250619.csv',sep=',',encoding='euc-kr')
print('---Data import---')
raw_copied = raw_data[raw_data['std_date'].isin([20240331,20240630,20240930,20241231])].copy()
null_columns = raw_copied.columns[raw_copied.isnull().any()]

print(f'null 값이 있는 컬럼 리스트 : {null_columns}')
for col in null_columns:
    new_column = f'{col}_isnull'
    raw_copied[new_column] = raw_copied[col].isnull().map(lambda x:'Null' if x else 'Notnull')
print('---IsNull 컬럼 생성 완료---')
object_dtype_list = raw_copied.select_dtypes(include='object').columns.tolist()
object_dtype_list.append('acct_titl_cd')
print(f'object 형식인 컬럼 리스트 : {object_dtype_list}')

for column in object_dtype_list:
    raw_copied[column] = raw_copied[column].astype('string')
print('---object 형식을 str 형식으로 변경 완료---')
raw_copied.drop(columns=['SSN_CORPNO','ACCT_KEY','acct_num','std_date','spcl_corp_new_mk'], inplace=True) # spcl_corp는 모든 value가 null이라서 drop
print(raw_copied.columns.tolist())

raw_copied['corp_estblsh_day'].unique()

from datetime import datetime

dt_list = ['STD_DATE_NEW','init_loan_dt','loan_deadln_dt']
dt_list_2 = ['corp_estblsh_day','init_regist_dt','lst_renew_dt']

for dt_col in dt_list:
    raw_copied[dt_col] = pd.to_datetime(raw_copied[dt_col].str.title(), format='%d%b%Y', errors='coerce') # Null 값은 그대로 놔둠
    print(f"{dt_col}이 날짜 형식으로 변환 완료되었습니다")
for dt_col in dt_list_2:
    raw_copied[dt_col] = pd.to_datetime(raw_copied[dt_col], format='%d%b%Y',errors='coerce') # Null 값은 그대로 놔둠
    print(f"{dt_col}이 날짜 형식으로 변환 완료되었습니다")

raw_copied['corp_estblsh_day'].fillna(pd.Timestamp('2024-01-01'), inplace=True)

reference = pd.Timestamp('2024-01-01')
print(reference)
print(type(reference))

raw_copied['cnt_days_since_0101'] = (raw_copied['STD_DATE_NEW'] - reference).dt.days
raw_copied['cnt_days_from_init'] = (raw_copied['STD_DATE_NEW'] - raw_copied['init_loan_dt']).dt.days
raw_copied['cnt_days_until_dead'] = (raw_copied['loan_deadln_dt'] - raw_copied['STD_DATE_NEW']).dt.days
raw_copied['cnt_days_total'] = (raw_copied['loan_deadln_dt'] - raw_copied['init_loan_dt']).dt.days
# raw_copied['cnt_days_estblsh'] = (raw_copied['STD_DATE_NEW'] - raw_copied['corp_estblsh_day']).dt.days
raw_copied['cnt_days_regist'] = (raw_copied['STD_DATE_NEW'] - raw_copied['init_regist_dt']).dt.days
raw_copied['cnt_days_renew'] = (raw_copied['STD_DATE_NEW'] - raw_copied['lst_renew_dt']).dt.days
print(raw_copied.info())

# data shrinkage implementation
raw_copied.drop(columns=dt_list,inplace=True)


raw_copied.drop(columns=['AF_CRG','AF_KIFRS_ADJ_NRML','AF_KIFRS_ADJ_SM','AF_KIFRS_ADJ_FXD_UNDER','AF_NRML_RATIO',\
                         'AF_SM_RATIO','AF_FXD_UNDER_RATIO','AF_ASSET_QUALITY','AF_DLNQNCY_DAY_CNT','AF_BNKRUT_CORP_FLG',\
                        'AF_BOND_ADJUST_FLG','AF_NO_PRFT_LOAN_MK','AF_NO_PRFT_LOAN_CAND_FLG','QUAL_CHG','NRML_CHG','SM_CHG','FXD_UNDER_CHG'],inplace=True)
raw_copied.drop(columns=['BF_CRG','BF_KIFRS_ADJ_NRML','BF_KIFRS_ADJ_SM','BF_KIFRS_ADJ_FXD_UNDER','BF_NRML_RATIO',\
                         'BF_SM_RATIO','BF_FXD_UNDER_RATIO','BF_ASSET_QUALITY','BF_DLNQNCY_DAY_CNT','BF_BNKRUT_CORP_FLG',\
                        'BF_BOND_ADJUST_FLG','BF_NO_PRFT_LOAN_MK','BF_NO_PRFT_LOAN_CAND_FLG'],inplace=True)


print(raw_copied.info())
print(raw_copied[['cnt_days_since_0101','cnt_days_from_init','cnt_days_until_dead','cnt_days_total']].head())

num_col_list = raw_copied.select_dtypes(include=['float64','int64']).columns.tolist()
str_col_list = raw_copied.select_dtypes(include='string').columns.tolist()

str_col_list_less_target = []
for col in str_col_list:
    if col != 'DLNQ_1M_FLG':
        str_col_list_less_target.append(col)

print(f"수치형 컬럼은 총 {len(num_col_list)}개 입니다")
print(raw_copied[num_col_list].info())
print()
print(f"범주형 컬럼은 총 {len(str_col_list_less_target)}개 입니다")
print(raw_copied[str_col_list_less_target].info())

raw_labeled_v1=raw_copied.copy()

for col in str_col_list:
    raw_labeled_v1[col][raw_labeled_v1[col].isnull()] = 'NA'

raw_labeled_v1['biz_gb'] = raw_labeled_v1['biz_gb'].str[:3]
print(raw_labeled_v1['biz_gb'].unique())

raw_labeled_v1['biz_gb'] = raw_labeled_v1['biz_gb'].str.lower()
for col in str_col_list:
    print(raw_labeled_v1[col].unique())


# LabelEncoder를 보관할 딕셔너리 (각 컬럼별로 학습된 encoder를 저장)
raw_labeled = raw_labeled_v1.copy()
encoders = {}
# 각 범주형 컬럼에 대해 LabelEncoder로 변환
for col in str_col_list:
    le = LabelEncoder()
    raw_labeled[col] = le.fit_transform(raw_labeled_v1[col])
    encoders[col] = le

nunique_str = {}
for i, c in enumerate(str_col_list_less_target):
    num_str = raw_labeled[c].max() + 1  # 라벨 인코딩 이후 각 컬럼별 value들 중 최대값 + 1 
    nunique_str[i] = num_str
print(nunique_str)

from sklearn.impute import SimpleImputer
import numpy as np

# 예시: 평균 대체
#imputer = SimpleImputer(strategy="median") # "mean", "median", "most_frequent", "constant" 옵션 존재
imputer = SimpleImputer(strategy="most_frequent")
# fit_transform으로 결측치 보간
num_data = raw_labeled[num_col_list].values 

num_data_imputed = imputer.fit_transform(num_data)
raw_labeled[num_col_list] = num_data_imputed
print('---결측치 최빈값으로 보간 완료---')
import pandas as pd
from sklearn.preprocessing import RobustScaler

# 로버스트 스케일러 초기화 (중앙값과 IQR을 사용)
scaler = RobustScaler()

raw_scaled = raw_labeled.copy()
raw_scaled[num_col_list] = scaler.fit_transform(raw_labeled[num_col_list])
print('---Robust Scaling completed---')
from sklearn.model_selection import train_test_split
# Step 1: train(70%) + temp(30%) 분할
train_df, temp_df = train_test_split(
    raw_scaled,
    test_size=0.3,
    random_state=42,
    stratify=raw_scaled["DLNQ_1M_FLG"]
)

# Step 2: temp → fine_tune(15%) + test(15%) 분할
fine_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,  # 50% of 20% = 10% of original
    random_state=42,
    stratify=temp_df["DLNQ_1M_FLG"]
)

# # Step 2: indi → fine_tune(70%) + test(30%) 분할
# fine_df, test_df = train_test_split(
#     raw_scaled_indi,
#     test_size=0.3,  
#     random_state=42,
#     stratify=raw_scaled_indi["DLNQ_1M_FLG"]
# )

X_train_str = train_df[str_col_list_less_target].values  # (행, 50)
X_train_num = train_df[num_col_list].values  # (행, 45)
y_train = train_df["DLNQ_1M_FLG"].values

X_fine_str = fine_df[str_col_list_less_target].values
X_fine_num = fine_df[num_col_list].values
y_fine = fine_df["DLNQ_1M_FLG"].values

X_test_str = test_df[str_col_list_less_target].values
X_test_num = test_df[num_col_list].values
y_test = test_df["DLNQ_1M_FLG"].values

print("Train size:", len(train_df), "Fine size:", len(fine_df),"Test size:", len(test_df))

from torch.utils.data import Dataset, DataLoader
import torch

class Data_loading(Dataset):
    def __init__(self, X_str, X_num, y):
        super().__init__()
        self.X_str = X_str
        self.X_num = X_num
        self.y = y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        str_feats = self.X_str[idx]  # shape: (51,)
        num_feats = self.X_num[idx]  # shape: (69,)
        label = self.y[idx]
        return {
            "str": torch.tensor(str_feats, dtype=torch.long),
            "num": torch.tensor(num_feats, dtype=torch.float),
            "label": torch.tensor(label, dtype=torch.long)
        }

train_dataset = Data_loading(X_train_str, X_train_num, y_train)
fine_dataset = Data_loading(X_fine_str, X_fine_num, y_fine)
test_dataset = Data_loading(X_test_str, X_test_num, y_test)

batch_size = 16  # 미니 배치 사이즈를 16로 설정
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
fine_loader = DataLoader(fine_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#device = torch.device("cpu")
classes = np.unique(y_train)  # y_train에 실제로 존재하는 모든 클래스 추출
class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score 
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# --- 기본 모델 구조 (원본 코드와 거의 동일) ---
# 이 부분은 V1, V2에서 공통으로 사용되는 인코더와 최종 분류기입니다.

class TabTransformerEncoder(nn.Module):
    """TabTransformer의 인코더 부분"""
    def __init__(self, num_cat_features, cat_max_dict, num_num_features,
                 d_model=32, nhead=4, num_layers=6, dim_feedforward=64, dropout_rate=0.3):
        super().__init__()
        self.d_model = d_model
        self.num_cat_features = num_cat_features
        self.num_num_features = num_num_features

        # 범주형 변수 임베딩
        self.embeddings = nn.ModuleList([
            nn.Embedding(cat_max_dict[i], d_model) for i in range(num_cat_features)
        ])
        # 수치형 변수 임베딩 (각 변수를 d_model 차원으로)
        self.numeric_embeddings = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(num_num_features)
        ])

        # Transformer 인코더 레이어 정의
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x_cat, x_num):
        # 범주형과 수치형 임베딩을 결합
        cat_emb = torch.cat([self.embeddings[i](x_cat[:, i]).unsqueeze(1)
                             for i in range(self.num_cat_features)], dim=1)
        num_emb = torch.cat([self.numeric_embeddings[i](x_num[:, i].unsqueeze(1)).unsqueeze(1)
                             for i in range(self.num_num_features)], dim=1)

        # (B, num_features, d_model) 형태의 시퀀스로 만듦
        x = torch.cat([cat_emb, num_emb], dim=1)

        # Transformer 통과
        x = self.transformer(x)
        return x

class TabTransformerClassifier(nn.Module):
    """사전학습된 인코더를 사용하는 최종 분류기"""
    def __init__(self, encoder, d_model=32, final_hidden=128, dropout_rate=0.3):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Sequential(
            nn.Linear(d_model, final_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(final_hidden, 2)
        )
    def forward(self, x_cat, x_num):
        encoded = self.encoder(x_cat, x_num)
        # 모든 변수(feature)의 출력을 평균내어 대표 벡터 생성
        pooled = encoded.mean(dim=1)
        return self.fc(pooled)

def mask_inputs(shape, mask_ratio, device):
    """
    주어진 shape과 mask_ratio에 따라 불리언 마스크를 생성합니다.
    True는 마스킹될 위치를 의미합니다.
    """
    B, L = shape
    return (torch.rand(B, L, device=device) < mask_ratio)


# ===============================================================================
#  V2: 대조 학습 (Contrastive Learning) 사전학습 # ===============================================================================

class ProjectionHead(nn.Module):
    """대조 학습을 위한 프로젝션 헤드"""
    def __init__(self, d_model, projection_dim=128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, projection_dim)
        )
    def forward(self, x):
        return self.head(x)

def tabular_augment(x_cat, x_num, mask_ratio=0.15):
    """
    테이블 데이터 증강 함수.
    입력 피처의 일부를 무작위로 마스킹(0으로 만듦)하여 증강된 뷰를 생성.
    """
    B, L_cat = x_cat.shape
    B, L_num = x_num.shape

    # 범주형 증강
    cat_mask = torch.rand(x_cat.shape, device=x_cat.device) < mask_ratio
    x_cat_aug = x_cat.clone()
    x_cat_aug[cat_mask] = 0 # 0은 보통 [PAD] 토큰으로 사용

    # 수치형 증강
    num_mask = torch.rand(x_num.shape, device=x_num.device) < mask_ratio
    x_num_aug = x_num.clone()
    x_num_aug[num_mask] = 0.0

    return x_cat_aug, x_num_aug

class NTXentLoss(nn.Module):
    """대조 학습을 위한 NT-Xent 손실 함수"""
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def forward(self, z_i, z_j):
        B, _ = z_i.shape
        # z_i와 z_j를 합쳐서 (2B, D) 텐서 생성
        z = torch.cat([z_i, z_j], dim=0)

        # 코사인 유사도 계산
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        # 자기 자신과의 유사도(대각선)를 제외하기 위한 마스크
        sim_i_j = torch.diag(sim, B)
        sim_j_i = torch.diag(sim, -B)

        # 긍정 쌍 (positive pairs)
        positive_samples = torch.cat([sim_i_j, sim_j_i], dim=0).reshape(2 * B, 1)

        # 자기 자신과의 비교를 제외한 부정 쌍 (negative pairs)
        mask = torch.ones((2 * B, 2 * B), dtype=bool, device=z.device)
        mask.fill_diagonal_(False)

        negative_samples = sim[mask].reshape(2 * B, -1)

        # 최종 로짓: [positive_sample, negative_samples]
        logits = torch.cat([positive_samples, negative_samples], dim=1)

        # 정답 레이블: 항상 0번째가 긍정 쌍이므로 0으로 채움
        labels = torch.zeros(2 * B, dtype=torch.long, device=z.device)

        return self.criterion(logits, labels)

def pretrain_contrastive_v2(encoder, projection_head, loader, epochs=10, device='cpu'):
    """V2: 대조 학습 사전학습 루프"""
    print("--- Starting V2: Contrastive Learning Pre-training ---")
    encoder.to(device).train()
    projection_head.to(device).train()

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(projection_head.parameters()), lr=3e-4
    )
    criterion = NTXentLoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            x_cat, x_num = batch["str"].to(device), batch["num"].to(device)

            # 1. 동일 데이터에 대해 두 개의 다른 증강된 뷰 생성
            x_cat_i, x_num_i = tabular_augment(x_cat, x_num)
            x_cat_j, x_num_j = tabular_augment(x_cat, x_num)

            # 2. 각 뷰를 인코더와 프로젝션 헤드에 통과
            h_i = encoder(x_cat_i, x_num_i).mean(dim=1) # (B, d_model)
            h_j = encoder(x_cat_j, x_num_j).mean(dim=1) # (B, d_model)

            z_i = projection_head(h_i) # (B, projection_dim)
            z_j = projection_head(h_j) # (B, projection_dim)

            # 3. NT-Xent 손실 계산
            loss = criterion(z_i, z_j)

            # 4. 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[Pretrain V2] Epoch {epoch+1}, Loss: {total_loss / len(loader):.4f}")


# ===============================================================================
#  미세 조정 (Fine-tuning) - 원본 코드와 동일
# ===============================================================================
def finetune(model, loader, test_loader, criterion, optimizer, device, epochs=10):
    print("\n--- Starting Fine-tuning ---")
    model.to(device)

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        all_labels, all_probs, all_preds = [], [], []

        for batch in loader:
            x_cat, x_num, y_true = batch["str"].to(device), batch["num"].to(device), batch["label"].to(device)
            logits = model(x_cat, x_num)
            loss = criterion(logits, y_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = (probs >= 0.5).long()
            all_labels.append(y_true.detach().cpu())
            all_probs.append(probs.detach().cpu())
            all_preds.append(preds.detach().cpu())

        y_true_all, y_prob_all, y_pred_all = torch.cat(all_labels), torch.cat(all_probs), torch.cat(all_preds)
        epoch_loss = total_loss / len(loader)
        epoch_acc = accuracy_score(y_true_all, y_pred_all)
        epoch_auc = roc_auc_score(y_true_all, y_prob_all)
        epoch_f1 = f1_score(y_true_all, y_pred_all)

        print(f"[Finetune] Epoch {epoch:2d} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | AUC: {epoch_auc:.4f} | F1: {epoch_f1:.4f}")

    # 최종 평가
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            cat_feats, num_feats, labels = batch["str"].to(device), batch["num"].to(device), batch["label"].to(device)
            logits = model(cat_feats, num_feats)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            y_prob.extend(probs[:, 1].cpu().tolist())

    acc = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred)
    print(f"   -> Test Accuracy: {acc:.4f}, Test ROC-AUC: {roc_auc:.4f}, Test F1-score: {f1:.4f}")

# --- V2: 대조 학습 실행 ---
encoder_v2 = TabTransformerEncoder(num_cat_features=len(str_col_list_less_target), cat_max_dict=nunique_str, num_num_features=len(num_col_list), d_model=32)
projection_head_v2 = ProjectionHead(d_model=32)
pretrain_contrastive_v2(encoder_v2, projection_head_v2, train_loader, epochs=1, device=device)


# V2 사전학습된 인코더로 미세조정
classifier_v2 = TabTransformerClassifier(encoder_v2, d_model=32)
optimizer_v2 = torch.optim.Adam(classifier_v2.parameters(), lr=1e-4)
criterion=nn.CrossEntropyLoss(weight=class_weights)
finetune(classifier_v2, fine_loader, test_loader, criterion, optimizer_v2, device, epochs=3)

# 가중치 저장은 미세조정 모델만 해도 괜찮지만, 추후 사용성을 높이기 위하여 pretraining 가중치도 저장할 것
# (1) pretraining weights 저장
torch.save(encoder_v2.state_dict(), 'pretrained_model_f.pth')
print('---pretrained_model_f.pth 저장 완료---')
# (2) fine-tuning weights 저장
torch.save(classifier_v2.state_dict(), 'finetuned_model_f.pth')
print('---finetuned_model_f.pth 저장 완료---')

# 모델 불러오기
# pretrained_model = TabTransformerEncoder() # 저장할 때와 동일한 클래스 구조
# pretrained_model.load_state_dict(torch.load('pretrained_model_f.pth'))
# pretrained_model.eval() # 추론 모드 돌입
# fine-tuned_model = TabTransformerClassifier() # 저장할 때와 동일한 클래스 구조
# fine-tuned_model.load_state_dict(torch.load('finetuned_model_f.pth'))
# fine-tuned_model.eval() # 추론 모드 돌입