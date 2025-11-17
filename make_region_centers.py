import pandas as pd

SRC_PATH = "bjd_info_except_boundary.csv"  # 원본 파일 이름

df = pd.read_csv(SRC_PATH, encoding="cp949")

print("원본 컬럼 목록:", df.columns.tolist())

cols = df.columns

# --- 1) 시도/시군구/위도/경도 컬럼 이름 찾기 ---

# 시도
if "시도" in cols:
    sido_col = "시도"
elif "sd_nm" in cols:
    sido_col = "sd_nm"
else:
    raise KeyError("시도 컬럼을 찾을 수 없습니다. (시도, sd_nm 둘 다 없음)")

# 시군구
if "시군구" in cols:
    sgg_col = "시군구"
elif "sgg_nm" in cols:
    sgg_col = "sgg_nm"
else:
    raise KeyError("시군구 컬럼을 찾을 수 없습니다. (시군구, sgg_nm 둘 다 없음)")

# 위도
if "center_lati" in cols:
    lat_col = "center_lati"
elif "위도" in cols:
    lat_col = "위도"
else:
    raise KeyError("위도 컬럼을 찾을 수 없습니다. (center_lati, 위도 둘 다 없음)")

# 경도
if "center_long" in cols:
    lon_col = "center_long"
elif "경도" in cols:
    lon_col = "경도"
else:
    raise KeyError("경도 컬럼을 찾을 수 없습니다. (center_long, 경도 둘 다 없음)")

if "level" not in cols:
    raise KeyError("'level' 컬럼이 없습니다. bjd_info_except_boundary.csv가 맞는지 확인하세요.")

# --- 2) 시군구 내부의 읍/면/동 좌표를 평균내서 중심 구하기 ---

# level 설명 (보통 이런 구조):
# 0: 시도, 1: 시군구, 2 이상: 읍/면/동/리 ...
# → 우리는 level >= 2 인 애들만 모아서 평균을 낼 거야.
sub = df[df["level"] >= 2].copy()

# 그룹별 평균 (시도, 시군구 단위)
grouped = (
    sub
    .groupby([sido_col, sgg_col])[[lat_col, lon_col]]
    .mean()
    .reset_index()
)

region_centers = grouped.rename(
    columns={
        sido_col: "시도",
        sgg_col: "시군구",
        lat_col: "lat",
        lon_col: "lon",
    }
)

# 공백 정리
region_centers["시도"] = region_centers["시도"].astype(str).str.strip()
region_centers["시군구"] = region_centers["시군구"].astype(str).str.strip()

OUT_PATH = "region_centers.csv"
region_centers.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

print("저장 완료:", OUT_PATH, "행 개수:", len(region_centers))
print(region_centers.head())
