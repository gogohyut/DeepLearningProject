import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt

# ===== 1. 설정 =====
DATA_FILES = {
    2013: "도로교통공단_일자별 시군구별 교통사고(2013).csv",
    2014: "도로교통공단_일자별 시군구별 교통사고(2014).csv",
    2015: "도로교통공단_일자별 시군구별 교통사고(2015).csv",
    2016: "도로교통공단_일자별 시군구별 교통사고(2016).csv",
    2017: "도로교통공단_일자별 시군구별 교통사고(2017).csv",
    2018: "도로교통공단_일자별 시군구별 교통사고(2018).csv",
    2019: "도로교통공단_일자별 시군구별 교통사고(2019).csv",
    2020: "도로교통공단_일자별 시군구별 교통사고(2020).csv",
    2021: "도로교통공단_일자별 시군구별 교통사고(2021).csv",
    2022: "도로교통공단_일자별 시군구별 교통사고(2022).csv",
    2023: "도로교통공단_일자별 시군구별 교통사고(2023).csv",
    2024: "도로교통공단_일자별 시군구별 교통사고(2024).csv",
}

MODEL_PATH = "accident_severity_mlp.h5"
RANDOM_STATE = 42

# ===== 2. 데이터 로드 =====
print("[INFO] 여러 연도 데이터 불러오는 중...")

dfs = []
for year, path in DATA_FILES.items():
    if not os.path.exists(path):
        print(f"  ! {year}년 파일 없음: {path} (건너뜀)")
        continue

    print(f"  - {year}년: {path}")
    df_year = pd.read_csv(path, encoding="cp949")

    # 파일 안에 '발생연도' 컬럼이 없으면, year로 채워 넣기
    if "발생연도" not in df_year.columns:
        df_year["발생연도"] = year

    dfs.append(df_year)

# 여러 연도 통합
df = pd.concat(dfs, ignore_index=True)

print("통합 데이터 shape:", df.shape)
print(df.head())
print("컬럼:", df.columns.tolist())

# ===== 2-1. 월별로 병합 (시도+시군구+발생월 단위) =====
# 연도 컬럼이 있으면 ['발생연도','발생월','시도','시군구'] 로 그룹하는 게 더 좋음
group_cols = ["발생월", "시도", "시군구"]

df_monthly = (
    df.groupby(group_cols)[["사고건수", "사망자수", "중상자수", "경상자수", "부상신고자수"]]
      .sum()
      .reset_index()
)

print("[INFO] 월별 집계 후 shape:", df_monthly.shape)
print(df_monthly.head())

# ===== 3. 월별 레이블 생성 (사상자 비율 기반) =====
def make_severity(row):
    # 사상자 합계
    minor = row["경상자수"] + row["부상신고자수"]
    serious = row["중상자수"]
    fatal = row["사망자수"]

    total = minor + serious + fatal

    # 사상자 없으면 그냥 경미로 처리
    if total == 0:
        return 0

    fatal_ratio = fatal / total
    serious_ratio = serious / total

    # ★ 임계값은 조정 가능: 일단 10%, 30%로 설정
    if fatal_ratio >= 0.10:
        return 2  # 사망 비율이 높은 달만 사망급
    elif serious_ratio >= 0.30:
        return 1  # 중상 비율이 높은 달
    else:
        return 0  # 나머지 -> 경미

df_monthly["severity"] = df_monthly.apply(make_severity, axis=1)

print("\n[레이블 분포(개수)]")
print(df_monthly["severity"].value_counts())
print("\n[레이블 분포(비율)]")
print(df_monthly["severity"].value_counts(normalize=True))

# ===== 4. 피처 선택 =====
# 이제 '발생일'은 완전히 안 씀
num_cols = ["발생월", "사고건수"]    # 필요하면 부상자수 합계도 추가 가능
cat_cols = ["시도", "시군구"]

for col in num_cols:
    df_monthly[col] = pd.to_numeric(df_monthly[col], errors="coerce")
df_monthly = df_monthly.dropna(subset=num_cols)

df_cat = pd.get_dummies(df_monthly[cat_cols], drop_first=False)

X_num = df_monthly[num_cols].reset_index(drop=True)
X_cat = df_cat.reset_index(drop=True)
X = pd.concat([X_num, X_cat], axis=1)

y = df_monthly["severity"].astype(int)

print("\n입력 피처 shape:", X.shape)

meta = {
    "feature_columns": X.columns.tolist(),
    "num_cols": num_cols,
    "cat_cols": cat_cols,
}
joblib.dump(meta, "accident_severity_meta.pkl")
print("[INFO] 메타 정보 저장 완료: accident_severity_meta.pkl")



# ===== 5. Train / Test 분할 =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y  # 클래스 비율 유지
)

# Train에서 다시 Train/Val 나누기
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y_train
)

print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

# ===== 6. 스케일링 =====
# (숫자 + 원핫 전체를 스케일링해도 크게 문제 없음)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

joblib.dump(scaler, "accident_severity_scaler.pkl")
print("[INFO] 스케일러 저장 완료: accident_severity_scaler.pkl")

input_dim = X_train_scaled.shape[1]
num_classes = len(np.unique(y_train))

print("입력 차원:", input_dim, "클래스 수:", num_classes)

# ===== 7. 클래스 불균형 처리 (class_weight) =====
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = {int(c): float(w) for c, w in zip(np.unique(y_train), class_weights)}
print("클래스 가중치:", class_weight_dict)

# ===== 8. MLP 모델 정의 =====
def build_mlp(input_dim, num_classes):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

model = build_mlp(input_dim, num_classes)
model.summary()

# ===== 9. 학습 =====
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )
]

history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=50,
    batch_size=256,
    class_weight=class_weight_dict,
    verbose=1
)

# ===== 9-1. 학습/검증 정확도 그래프 =====
# Keras history 객체에서 accuracy, val_accuracy 가져오기
acc = history.history.get("accuracy", history.history.get("acc"))
val_acc = history.history.get("val_accuracy", history.history.get("val_acc"))

if acc is not None and val_acc is not None:
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, acc, marker="o", label="Train Acc")
    plt.plot(epochs_range, val_acc, marker="o", label="Val Acc")
    plt.title("Training / Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    # 파일로도 저장 (보고서에 쓰기 편하게)
    plt.savefig("training_accuracy.png", dpi=150, bbox_inches="tight")
    print("[INFO] 학습 정확도 그래프 저장: training_accuracy.png")

    # VSCode에서 바로 보고 싶으면 주석 해제
    plt.show()
else:
    print("[WARN] history에 accuracy 정보가 없습니다. metrics 설정을 확인하세요.")

# ===== 10. 평가 =====
print("\n[INFO] 테스트 데이터 평가")
test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

y_pred_proba = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_proba, axis=1)

print("\n[Classification Report]")
print(classification_report(y_test, y_pred, digits=4))

print("\n[Confusion Matrix]")
print(confusion_matrix(y_test, y_pred))

# ===== 11. 모델 저장 =====
model.save(MODEL_PATH)
print(f"\n[INFO] 모델 저장 완료: {MODEL_PATH}")

# ===== 지역·월별 "사상자 수 기준" 심각도 비율 통계 저장 =====
# df_monthly에는 이미 연도+월+시군구별 합계가 들어있으므로
# 여기서 다시 연도를 합쳐서 2013~2024 전체 합계를 만든다.

stats = (
    df_monthly
    .groupby(["발생월", "시도", "시군구"])[["사망자수", "중상자수", "경상자수", "부상신고자수"]]
    .sum()
    .reset_index()
)

# 경미 / 중상 / 사망급 사상자 수
stats["cnt_minor"] = stats["경상자수"] + stats["부상신고자수"]
stats["cnt_serious"] = stats["중상자수"]
stats["cnt_fatal"] = stats["사망자수"]

# 전체 사상자 수
stats["total_cnt"] = stats[["cnt_minor", "cnt_serious", "cnt_fatal"]].sum(axis=1)

# 0~1 비율
stats["ratio_0"] = stats["cnt_minor"] / stats["total_cnt"]
stats["ratio_1"] = stats["cnt_serious"] / stats["total_cnt"]
stats["ratio_2"] = stats["cnt_fatal"] / stats["total_cnt"]

stats.to_csv("severity_stats_by_region_month.csv", index=False, encoding="utf-8-sig")
print("[INFO] 지역·월별 사상자 수 기준 심각도 비율 통계 저장 완료: severity_stats_by_region_month.csv")
