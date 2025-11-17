import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

MODEL_PATH = "accident_severity_mlp.h5"
SCALER_PATH = "accident_severity_scaler.pkl"
META_PATH = "accident_severity_meta.pkl"

# 0: 경미, 1: 중상, 2: 사망급
SEVERITY_LABELS = {
    0: "경미 사고",
    1: "중상 사고",
    2: "사망급 사고"
}

def load_assets():
    print("[INFO] 모델 / 스케일러 / 메타 로딩 중...")
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    meta = joblib.load(META_PATH)
    feature_columns = meta["feature_columns"]
    return model, scaler, feature_columns

def build_feature_row(feature_columns, month, day, accident_cnt,
                      sido, sigungu):
    # 모든 피처 0으로 초기화
    data = {col: 0 for col in feature_columns}

    # 숫자형 피처
    if "발생월" in data:
        data["발생월"] = month
    if "발생일" in data:
        data["발생일"] = day
    if "사고건수" in data:
        data["사고건수"] = accident_cnt

    # 원-핫 범주형 피처
    sido_col = f"시도_{sido}"
    sigungu_col = f"시군구_{sigungu}"

    if sido_col in data:
        data[sido_col] = 1
    else:
        print(f"[WARN] 학습 데이터에 없는 시도 값입니다: {sido} (컬럼 {sido_col} 없음)")

    if sigungu_col in data:
        data[sigungu_col] = 1
    else:
        print(f"[WARN] 학습 데이터에 없는 시군구 값입니다: {sigungu} (컬럼 {sigungu_col} 없음)")

    df_row = pd.DataFrame([data], columns=feature_columns)
    return df_row

def predict_severity(model, scaler, feature_columns,
                     month, day, accident_cnt,
                     sido, sigungu):
    X_row = build_feature_row(feature_columns,
                              month, day, accident_cnt,
                              sido, sigungu)
    X_scaled = scaler.transform(X_row)
    proba = model.predict(X_scaled)[0]   # (3,)
    pred_class = int(np.argmax(proba))
    return pred_class, proba

if __name__ == "__main__":
    # 1. 자산 로딩
    model, scaler, feature_columns = load_assets()

    print("\n=== 교통사고 심각도 예측 데모 ===")
    print("※ 시도/시군구 이름은 CSV에 나오는 것과 똑같이 입력해야 합니다.")
    print("   예) 서울특별시, 부산광역시 / 강남구, 중구 등\n")

    # 2. 입력 받기
    month = int(input("발생월(1~12): "))
    day = int(input("발생일(1~31): "))
    accident_cnt = int(input("사고건수(보통 1 입력): "))

    sido = input("시도 이름 (예: 서울특별시): ").strip()
    sigungu = input("시군구 이름 (예: 강남구): ").strip()

    # 3. 예측
    pred_class, proba = predict_severity(
        model, scaler, feature_columns,
        month, day, accident_cnt,
        sido, sigungu
    )

    print("\n=== 예측 결과 ===")
    print(f"예측 심각도: {SEVERITY_LABELS[pred_class]}")
    print("각 클래스별 확률:")
    print(f"  경미 사고(0): {proba[0]:.3f}")
    print(f"  중상 사고(1): {proba[1]:.3f}")
    print(f"  사망급 사고(2): {proba[2]:.3f}")
