# -*- coding: utf-8 -*-
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import DATA_DIR, RESULTS_DIR
"""XGBoost와 CatBoost에 동일한 탐색 예산을 주고 공정 비교한다.

프로토콜
  탐색  : fold seed 0 의 5겹으로만 45회 무작위 탐색 -> 최적 파라미터 1개 선택
  평가  : 탐색에 쓰지 않은 fold seed 1,2,3,4 로 성능 측정
  이유  : 파라미터를 고른 분할과 성능을 보고하는 분할을 분리해야 선택 편향이 없다.

라이브러리를 하나씩만 import 한다 (OpenMP 충돌 회피).
사용법: python exp11.py <XGBoost|CatBoost>
"""
import sys, json, time
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

gm = raw.groupby("단지키")["층"].transform("max")
raw["단지최고층"] = gm
raw["층비율"] = (raw["층"] / gm.replace(0, np.nan)).fillna(0.5)
raw["단지평균면적"] = raw.groupby("단지키")["전용면적(㎡)"].transform("mean")
raw["재건축연한"] = (raw["건축년도"] <= 1989).astype(int)
raw["시청거리(m)"] = hav(raw["단지 위도"], raw["단지 경도"], 37.5665, 126.9780)
raw["강남역거리(m)"] = hav(raw["단지 위도"], raw["단지 경도"], 37.4979, 127.0276)
NEW = ["단지최고층", "층비율", "단지평균면적", "재건축연한", "시청거리(m)", "강남역거리(m)",
       "단지 위도", "단지 경도"]

y = np.log(raw["거래금액(만원)"] / raw["전용면적(㎡)"]).values
X = pd.concat([raw[BASE + NEW], pd.get_dummies(raw["구"], prefix="구")], axis=1).astype(float)
X.columns = ["f%d" % i for i in range(X.shape[1])]
Xv = X.values

keys = raw["단지키"].unique()
def folds_for(seed, k=5):
    rng = np.random.default_rng(seed)
    assign = dict(zip(keys, rng.integers(0, k, len(keys))))
    f = raw["단지키"].map(assign).values
    return [(np.where(f != i)[0], np.where(f == i)[0]) for i in range(k)]

def sample(rng):
    if MODEL == "XGBoost":
        return dict(max_depth=int(rng.integers(3, 11)),
                    learning_rate=float(np.exp(rng.uniform(np.log(0.01), np.log(0.3)))),
                    n_estimators=int(rng.integers(150, 1000)),
                    subsample=float(rng.uniform(0.6, 1.0)),
                    colsample_bytree=float(rng.uniform(0.5, 1.0)),
                    min_child_weight=int(rng.integers(1, 20)),
                    reg_lambda=float(np.exp(rng.uniform(np.log(0.5), np.log(30)))),
                    reg_alpha=float(np.exp(rng.uniform(np.log(1e-3), np.log(5)))),
                    gamma=float(rng.uniform(0, 4)))
    return dict(depth=int(rng.integers(4, 11)),
                learning_rate=float(np.exp(rng.uniform(np.log(0.01), np.log(0.3)))),
                iterations=int(rng.integers(150, 1000)),
                l2_leaf_reg=float(np.exp(rng.uniform(np.log(0.5), np.log(30)))),
                random_strength=float(rng.uniform(0, 3)),
                bagging_temperature=float(rng.uniform(0, 2)))

def build(p, seed):
    if MODEL == "XGBoost":
        import xgboost as xgb
        return xgb.XGBRegressor(objective="reg:squarederror", random_state=seed,
                                tree_method="hist", n_jobs=4, **p)
    from catboost import CatBoostRegressor
    return CatBoostRegressor(random_seed=seed, verbose=0, thread_count=4,
                             allow_writing_files=False, **p)

def oof_mae(p, seed):
    o = np.zeros(len(Xv))
    for tr, te in folds_for(seed):
        m = build(p, seed)
        m.fit(Xv[tr], y[tr])
        o[te] = m.predict(Xv[te])
    return mean_absolute_error(np.exp(y), np.exp(o)), r2_score(y, o)

# ---- 탐색: seed 0 에서만 ----
t0 = time.time()
rng = np.random.default_rng(42)
trials = []
for i in range(45):
    p = sample(rng)
    mae, _ = oof_mae(p, 0)
    trials.append((mae, p))
trials.sort(key=lambda t: t[0])
best_mae_s0, best = trials[0]
search_sec = time.time() - t0

# ---- 평가: seed 1~4 ----
ev = [oof_mae(best, s) for s in [1, 2, 3, 4]]
maes = np.array([e[0] for e in ev]); r2s = np.array([e[1] for e in ev])

res = {"model": MODEL, "trials": 45, "search_seconds": round(search_sec, 1),
       "best_params": best, "search_mae_seed0": best_mae_s0,
       "eval_mae_mean": float(maes.mean()), "eval_mae_std": float(maes.std(ddof=1)),
       "eval_r2_mean": float(r2s.mean()), "eval_r2_std": float(r2s.std(ddof=1)),
       "eval_mae_all": [float(v) for v in maes],
       "search_mae_min": trials[0][0], "search_mae_median": trials[len(trials) // 2][0],
       "search_mae_max": trials[-1][0]}
print("%-9s 탐색 45회 %.0f초 | 선택 파라미터의 평가 MAE %.1f (±%.2f) | R2 %.4f"
      % (MODEL, search_sec, res["eval_mae_mean"], res["eval_mae_std"], res["eval_r2_mean"]))
print("          탐색 분포(seed0) 최저 %.1f / 중앙 %.1f / 최고 %.1f"
      % (res["search_mae_min"], res["search_mae_median"], res["search_mae_max"]))
print("          선택 파라미터:", {k: (round(v, 4) if isinstance(v, float) else v)
                              for k, v in best.items()})
P = RESULTS_DIR
json.dump(res, open(P + "\\fair_%s.json" % MODEL, "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
