# -*- coding: utf-8 -*-
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR
"""공통 데이터 파이프라인.

설계 원칙
  - 종속변수는 제곱미터당 단가. 거래 총액은 면적과의 곱이라 항등식이 된다.
  - 시계열 파생변수는 t분기 예측에 t-1 이전 분기만 쓴다.
  - 피처를 계보별로 나눠 각 계보의 한계 기여를 잴 수 있게 한다.
"""
import numpy as np, pandas as pd

B = DATA_DIR

INF = ["편의점(개)", "약국(개)", "병원(개)", "공공도서관(개)", "경찰서(개)", "소방서(개)",
       "대형마트(개)", "초등학교 거리(m)", "중학교 거리(m)", "고등학교 거리(m)",
       "지하철역까지 거리(m)", "정류소까지 거리(m)"]
PHYS = ["전용면적(㎡)", "층", "건축년도"]
GEO = ["단지 위도", "단지 경도", "시청거리(m)", "강남역거리(m)"]
CPLX = ["단지최고층", "층비율", "단지평균면적", "단지거래수"]
TIME = ["lag1", "lag_ma4", "모멘텀", "lag_vol", "구_lag1", "구_모멘텀"]

CFG = {
    "아파트 전세": ("아파트_전월세.csv", "전세", "보증금(만원)"),
    "아파트 월세": ("아파트_전월세.csv", "월세", "보증금(만원)"),
    "연립 매매":  ("연립다세대_매매.csv", None, "거래금액(만원)"),
    "연립 전세":  ("연립다세대_전월세.csv", "전세", "보증금(만원)"),
    "연립 월세":  ("연립다세대_전월세.csv", "월세", "보증금(만원)"),
}
_KEEP = (["시군구", "단지명", "년도", "분기", "거래금액(만원)", "보증금(만원)",
          "월세금(만원)", "전월세구분", "거리(m)", "최단거리(m)", "단지 위도", "단지 경도"]
         + PHYS + [c for c in INF if c not in ("지하철역까지 거리(m)", "정류소까지 거리(m)")])


def _num(s):
    if s.dtype == object:
        s = s.astype(str).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(s, errors="coerce")


def _hav(la1, lo1, la2, lo2):
    R = 6371.0
    p1, p2 = np.radians(la1), np.radians(la2)
    dp, dl = np.radians(la2 - la1), np.radians(lo2 - lo1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)) * 1000


def load(segname, need_geo=True):
    f, gubun, tgt = CFG[segname]
    d = pd.read_csv(B + "\\" + f, encoding="CP949", low_memory=False,
                    usecols=lambda c: c in _KEEP)
    d = d.rename(columns={"거리(m)": "지하철역까지 거리(m)",
                          "최단거리(m)": "정류소까지 거리(m)"})
    if gubun:
        d = d[d["전월세구분"] == gubun]
    sp = d["시군구"].str.split()
    d["구"] = sp.str[1]; d["동"] = sp.str[2]
    d["구동"] = d["구"] + "_" + d["동"]
    d["단지키"] = d["구동"] + "_" + d["단지명"].astype(str)

    cols = PHYS + INF + [tgt, "년도", "분기"] + (["단지 위도", "단지 경도"] if need_geo else [])
    for c in cols:
        d[c] = _num(d[c])
    req = PHYS + INF + [tgt, "구동"] + (["단지 위도", "단지 경도"] if need_geo else [])
    d = d.dropna(subset=req)
    d = d[(d[tgt] > 0) & (d["전용면적(㎡)"] > 0)].reset_index(drop=True)

    d["t"] = (d["년도"] - 2019) * 4 + d["분기"]
    d = d[(d["t"] >= 1) & (d["t"] <= 20)].reset_index(drop=True)
    d["단가"] = d[tgt] / d["전용면적(㎡)"]

    # 공간
    if need_geo:
        d["시청거리(m)"] = _hav(d["단지 위도"], d["단지 경도"], 37.5665, 126.9780)
        d["강남역거리(m)"] = _hav(d["단지 위도"], d["단지 경도"], 37.4979, 127.0276)

    # 단지 집계 (예측 시점에 관측 가능한 건물 속성)
    g = d.groupby("단지키")
    mx = g["층"].transform("max")
    d["단지최고층"] = mx
    d["층비율"] = (d["층"] / mx.replace(0, np.nan)).fillna(0.5)
    d["단지평균면적"] = g["전용면적(㎡)"].transform("mean")
    d["단지거래수"] = g["단가"].transform("size")

    # 시계열 : t 시점에는 t-1 이전만
    d = _add_lag(d)
    d["tgt_col"] = tgt
    return d, tgt


def _add_lag(d):
    a = d.groupby(["구동", "t"])["단가"].agg(["median", "size"]).reset_index()
    a.columns = ["구동", "t", "m", "n"]
    full = pd.MultiIndex.from_product([a["구동"].unique(), range(1, 21)],
                                      names=["구동", "t"]).to_frame(index=False)
    a = full.merge(a, on=["구동", "t"], how="left").sort_values(["구동", "t"])
    gg = a.groupby("구동", sort=False)
    a["lag1"] = gg["m"].shift(1)
    a["lag_ma4"] = (gg["m"].shift(1)
                    .rolling(4, min_periods=1).median().reset_index(level=0, drop=True))
    a["lag_vol"] = gg["n"].shift(1)
    a["모멘텀"] = a["lag1"] / a["lag_ma4"] - 1

    b = d.groupby(["구", "t"])["단가"].median().reset_index().rename(columns={"단가": "gm"})
    gfull = pd.MultiIndex.from_product([b["구"].unique(), range(1, 21)],
                                       names=["구", "t"]).to_frame(index=False)
    b = gfull.merge(b, on=["구", "t"], how="left").sort_values(["구", "t"])
    bg = b.groupby("구", sort=False)
    b["구_lag1"] = bg["gm"].shift(1)
    b["구_ma4"] = (bg["gm"].shift(1)
                  .rolling(4, min_periods=1).median().reset_index(level=0, drop=True))
    b["구_모멘텀"] = b["구_lag1"] / b["구_ma4"] - 1
    b = b.drop(columns=["gm", "구_ma4"])

    # 전체 시장 lag : 동도 구도 비었을 때의 최후 대체값
    mkt = d.groupby("t")["단가"].median().reindex(range(1, 21))
    mkt_lag = mkt.shift(1).ffill().bfill()

    d = d.merge(a[["구동", "t", "lag1", "lag_ma4", "lag_vol", "모멘텀"]],
                on=["구동", "t"], how="left")
    d = d.merge(b, on=["구", "t"], how="left")
    d["시장_lag"] = d["t"].map(mkt_lag)
    for c in ["lag1", "lag_ma4"]:
        d[c] = d[c].fillna(d["구_lag1"]).fillna(d["시장_lag"])
    d["구_lag1"] = d["구_lag1"].fillna(d["시장_lag"])
    d["lag_vol"] = d["lag_vol"].fillna(0)
    d["모멘텀"] = d["모멘텀"].fillna(0)
    d["구_모멘텀"] = d["구_모멘텀"].fillna(0)
    return d


def families(d, with_geo=True):
    """피처 계보. 누적으로 쌓아 각 계보의 한계 기여를 잰다."""
    gu = pd.get_dummies(d["구"], prefix="구").astype(float)
    fam = [("F0 물건기본", PHYS),
           ("F1 +인프라12", PHYS + INF),
           ("F2 +구더미", PHYS + INF),
           ("F3 +공간연속", PHYS + INF + (GEO if with_geo else [])),
           ("F4 +단지집계", PHYS + INF + (GEO if with_geo else []) + CPLX),
           ("F5 +시계열", PHYS + INF + (GEO if with_geo else []) + CPLX + TIME)]
    out = []
    for i, (nm, cols) in enumerate(fam):
        X = d[cols].astype(float)
        if i >= 2:
            X = pd.concat([X, gu], axis=1)
        names = list(X.columns)
        X.columns = ["x%d" % j for j in range(X.shape[1])]
        out.append((nm, X, names))
    return out
