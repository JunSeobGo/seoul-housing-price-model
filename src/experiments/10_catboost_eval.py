# -*- coding: utf-8 -*-
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import DATA_DIR, RESULTS_DIR
"""모델 하나만 별도 프로세스에서 평가한다.
XGBoost와 LightGBM을 같은 프로세스에 올리면 OpenMP 런타임이 충돌해
접근 위반으로 죽으므로, 라이브러리를 하나씩만 import 한다.
사용법: python exp10.py <LightGBM|CatBoost>
"""
import sys, json
import numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

MODEL = sys.argv[1]
B = DATA_DIR

raw = pd.read_csv(B + r"\아파트_매매.csv", encoding="CP949", low_memory=False)
raw = raw[(raw["년도"] == 2019) & (raw["분기"] == 1)].copy().reset_index(drop=True)
sp = raw["시군구"].str.split()
raw["구"] = sp.str[1]; raw["동"] = sp.str[2]
raw = raw.rename(columns={"거리(m)": "지하철역까지 거리(m)", "최단거리(m)": "정류소까지 거리(m)"})
raw["거래금액(만원)"] = pd.to_numeric(
    raw["거래금액(만원)"].astype(str).str.replace(",", "", regex=False), errors="coerce")
raw["단지키"] = raw["구"] + "_" + raw["동"] + "_" + raw["단지명"].astype(str)

BASE = ["전용면적(㎡)", "층", "건축년도", "편의점(개)", "약국(개)", "병원(개)", "공공도서관(개)",
        "경찰서(개)", "소방서(개)", "대형마트(개)", "초등학교 거리(m)", "중학교 거리(m)",
        "고등학교 거리(m)", "지하철역까지 거리(m)", "정류소까지 거리(m)"]
for c in BASE + ["단지 위도", "단지 경도"]:
    raw[c] = pd.to_numeric(raw[c], errors="coerce")
raw = raw.dropna(subset=BASE + ["거래금액(만원)", "단지 위도", "단지 경도"]).reset_index(drop=True)

def hav(la1, lo1, la2, lo2):
    R = 6371.0
    p1, p2 = np.radians(la1), np.radians(la2)
    dp, dl = np.radians(la2 - la1), np.radians(lo2 - lo1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)) * 1000

grpmax = raw.groupby("단지키")["층"].transform("max")
raw["단지최고층"] = grpmax
raw["층비율"] = (raw["층"] / grpmax.replace(0, np.nan)).fillna(0.5)
raw["단지평균면적"] = raw.groupby("단지키")["전용면적(㎡)"].transform("mean")
raw["재건축연한"] = (raw["건축년도"] <= 1989).astype(int)
raw["시청거리(m)"] = hav(raw["단지 위도"], raw["단지 경도"], 37.5665, 126.9780)
raw["강남역거리(m)"] = hav(raw["단지 위도"], raw["단지 경도"], 37.4979, 127.0276)
NEW = ["단지최고층", "층비율", "단지평균면적", "재건축연한", "시청거리(m)", "강남역거리(m)",
       "단지 위도", "단지 경도"]

y = np.log(raw["거래금액(만원)"] / raw["전용면적(㎡)"]).values
gu = pd.get_dummies(raw["구"], prefix="구")
X = pd.concat([raw[BASE + NEW], gu], axis=1).astype(float)
X.columns = ["f%d" % i for i in range(X.shape[1])]

keys = raw["단지키"].unique()
def folds_for(seed, k=5):
    rng = np.random.default_rng(seed)
    assign = dict(zip(keys, rng.integers(0, k, len(keys))))
    f = raw["단지키"].map(assign).values
    return [(np.where(f != i)[0], np.where(f == i)[0]) for i in range(k)]

def build(seed):
    if MODEL == "LightGBM":
        import lightgbm as lgb
        return lgb.LGBMRegressor(random_state=seed, n_estimators=600, learning_rate=0.05,
                                 num_leaves=31, min_child_samples=20, subsample=0.8,
                                 subsample_freq=1, colsample_bytree=0.7, n_jobs=1, verbose=-1)
    if MODEL == "CatBoost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(random_seed=seed, iterations=800, learning_rate=0.05,
                                 depth=6, l2_leaf_reg=3.0, verbose=0, thread_count=4,
                                 allow_writing_files=False)
    raise ValueError(MODEL)

maes, r2s = [], []
for s in [0, 1, 2, 3, 4]:
    o = np.zeros(len(X))
    for tr, te in folds_for(s):
        m = build(s)
        m.fit(X.iloc[tr].values, y[tr])
        o[te] = m.predict(X.iloc[te].values)
    maes.append(mean_absolute_error(np.exp(y), np.exp(o)))
    r2s.append(r2_score(y, o))

maes, r2s = np.array(maes), np.array(r2s)
res = {"model": MODEL, "mae_mean": float(maes.mean()), "mae_std": float(maes.std(ddof=1)),
       "r2_mean": float(r2s.mean()), "r2_std": float(r2s.std(ddof=1)),
       "mae_all": [float(v) for v in maes]}
print("%-12s MAE %.1f (±%.2f)  R2 %.4f (±%.4f)"
      % (MODEL, res["mae_mean"], res["mae_std"], res["r2_mean"], res["r2_std"]))
P = RESULTS_DIR
json.dump(res, open(P + "\\res_%s.json" % MODEL, "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
