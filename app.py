import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pydeck as pdk
import tensorflow as tf

# ===== 경로 설정 =====
MODEL_PATH = "accident_severity_mlp.h5"
SCALER_PATH = "accident_severity_scaler.pkl"
META_PATH = "accident_severity_meta.pkl"
REGION_PATH = "region_centers.csv"
STATS_PATH = "severity_stats_by_region_month.csv"   # train 스크립트에서 새로 만든 통계 파일

SEVERITY_LABELS = {
    0: "경미 사고",
    1: "중상 사고",
    2: "사망급 사고"
}

# ===== 시도 이름 매핑 (regions vs stats) =====
def normalize_sido_for_stats(sido: str) -> str:
    """regions에서 쓰는 시도 이름을 통계(stats)에서 쓰는 이름으로 변환"""
    mapping = {
        "서울특별시": "서울",
        "부산광역시": "부산",
        "대구광역시": "대구",
        "인천광역시": "인천",
        "광주광역시": "광주",
        "대전광역시": "대전",
        "울산광역시": "울산",
        "세종특별자치시": "세종",
        "경기도": "경기",
        "강원도": "강원",
        "충청북도": "충북",
        "충청남도": "충남",
        "전라북도": "전북",
        "전라남도": "전남",
        "경상북도": "경북",
        "경상남도": "경남",
        "제주특별자치도": "제주",
    }
    return mapping.get(sido, sido)


# ===== 모델 / 스케일러 / 메타 로딩 =====
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    meta = joblib.load(META_PATH)
    feature_columns = meta["feature_columns"]
    return model, scaler, feature_columns

@st.cache_data
def load_regions():
    return pd.read_csv(REGION_PATH)

@st.cache_data
def load_stats():
    """2013~2024 월·지역별 '사상자 수 기준' 심각도 비율 통계 로딩"""
    df = pd.read_csv(STATS_PATH)

    # 타입/공백 정리
    df["발생월"] = pd.to_numeric(df["발생월"], errors="coerce").astype("Int64")
    df["시도"] = df["시도"].astype(str).str.strip()
    df["시군구"] = df["시군구"].astype(str).str.strip()

    return df


# ===== 입력 피처 한 행 만들기 =====
def build_feature_row(feature_columns, month, sido, sigungu):
    # 모든 피처를 0으로 초기화
    data = {col: 0 for col in feature_columns}

    # 숫자 피처
    if "발생월" in data:
        data["발생월"] = month

    # 학습에서 사고건수를 피처에 넣었다면 1로 고정
    if "사고건수" in data:
        data["사고건수"] = 1

    # 원-핫 인코딩된 시도 / 시군구 컬럼 설정
    sido_col = f"시도_{sido}"
    sigungu_col = f"시군구_{sigungu}"

    if sido_col in data:
        data[sido_col] = 1
    if sigungu_col in data:
        data[sigungu_col] = 1

    # 모델 입력 형태로 변환 (1행짜리 DataFrame)
    X_row = pd.DataFrame([data])[feature_columns]
    return X_row


def predict_severity(model, scaler, feature_columns, month, sido, sigungu):
    X_row = build_feature_row(feature_columns, month, sido, sigungu)
    X_scaled = scaler.transform(X_row)
    proba = model.predict(X_scaled, verbose=0)[0]
    pred_class = int(np.argmax(proba))
    return pred_class, proba


# ===== Streamlit UI 설정 =====
st.set_page_config(page_title="교통사고 심각도 예측", layout="wide")

model, scaler, feature_columns = load_assets()
regions = load_regions()
stats = load_stats()

st.title("교통사고 심각도 예측 웹(Demo.Ver)") 

st.markdown("""
월과 시·군·구를 선택하면, 해당 지역에서 **사고가 발생했을 때**  
경미 / 중상 / 사망급 사고가 어느 쪽에 더 가까운지 MLP 모델이 예측합니다.  

또한 2013년~2024년 한국도로교통공단 통계를 바탕으로,  
해당 지역·해당 월에 발생한 **사상자 수 기준 경미 / 중상 / 사망 사고 비율**도 함께 보여줍니다.
""")

# --- 사이드바: 입력 ---
with st.sidebar:
    st.header("입력값 설정")

    month = st.selectbox("사고 발생 월", list(range(1, 13)), index=5)  # 기본값 6월
    sido_list = sorted(regions["시도"].unique())
    sido = st.selectbox("시도 선택", sido_list)

    sigungu_list = sorted(regions[regions["시도"] == sido]["시군구"].unique())
    sigungu = st.selectbox("시군구 선택", sigungu_list)

    run = st.button("예측하기")


# --- 메인 영역 ---
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("예측 결과")

    if run:
        # 1) MLP 예측 결과
        pred_class, proba = predict_severity(
            model, scaler, feature_columns,
            month, sido, sigungu
        )

        st.markdown(f"**예측 심각도:** {SEVERITY_LABELS[pred_class]}")
        st.write("각 클래스별 예측 확률:")

        prob_df = pd.DataFrame({
            "class": ["경미 사고(0)", "중상 사고(1)", "사망급 사고(2)"],
            "prob": proba
        })
        st.bar_chart(prob_df.set_index("class"))

        # 2) 2013~2024 실제 통계 비율 (사상자 수 기준)
        stats_sido = normalize_sido_for_stats(sido)

        row = stats[
            (stats["발생월"] == month) &
            (stats["시도"] == stats_sido) &
            (stats["시군구"] == sigungu)
        ]

        if not row.empty:
            r = row.iloc[0]

            # 퍼센트(0~100)로 변환
            percent_0 = float(r["ratio_0"]) * 100
            percent_1 = float(r["ratio_1"]) * 100
            percent_2 = float(r["ratio_2"]) * 100

            hist_df = pd.DataFrame({
                "class": ["경미 사고(0)", "중상 사고(1)", "사망급 사고(2)"],
                "percent": [percent_0, percent_1, percent_2],
            })

            st.markdown("---")
            st.markdown("#### 2013~2024 실제 통계 기준 비율 (사상자 수 기준, %)")

            st.bar_chart(hist_df.set_index("class"))

            st.caption(
                f"2013~2024년 동안 {sido} {sigungu} {month}월에 발생한 사상자 수 기준 비율입니다.\n"
                f"- 경미(경상+부상신고): {int(r['cnt_minor'])}명  "
                f"- 중상: {int(r['cnt_serious'])}명  "
                f"- 사망: {int(r['cnt_fatal'])}명  "
                f"(총 {int(r['total_cnt'])}명)"
            )
        else:
            st.info("해당 지역·월에 대한 통계 데이터가 부족합니다.")

        st.caption(f"선택한 월: {month}월, 지역: {sido} {sigungu}")


with col_right:
    st.subheader("위치")

    selected_region = regions[(regions["시도"] == sido) & (regions["시군구"] == sigungu)]

    if not selected_region.empty:
        lat = float(selected_region.iloc[0]["lat"])
        lon = float(selected_region.iloc[0]["lon"])

        # 구 중심 좌표 하나짜리 데이터프레임
        map_df = pd.DataFrame({"lat": [lat], "lon": [lon]})

        # 반지름 큰 원(버블)로 구 영역 느낌 내기 (단위: m)
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position="[lon, lat]",
            get_radius=3000,
            get_fill_color=[255, 0, 0, 60],   # 안쪽 색 (R,G,B,투명도)
            get_line_color=[255, 0, 0, 200],  # 테두리 색
            pickable=False,
        )

        view_state = pdk.ViewState(
            latitude=lat,
            longitude=lon,
            zoom=11,
            pitch=0,
            bearing=0,
        )

        deck = pdk.Deck(
            initial_view_state=view_state,
            layers=[layer],
        )

        st.pydeck_chart(deck)
    else:
        st.info("좌표 정보가 없는 시군구입니다. region_centers.csv를 확인하세요.")
