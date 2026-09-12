# -*- coding: utf-8 -*-
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import DATA_DIR, RESULTS_DIR
import numpy as np, pandas as pd, xgboost as xgb, lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

OUT = []
def say(s=""):
    OUT.append(str(s)); print(s)

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

# ---- 신규 파생변수 ----
def hav(la1, lo1, la2, lo2):
    R = 6371.0
    p1, p2 = np.radians(la1), np.radians(la2)
    dp, dl = np.radians(la2 - la1), np.radians(lo2 - lo1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)) * 1000

grpmax = raw.groupby("단지키")["층"].transform("max")
raw["단지최고층"] = grpmax
raw["층비율"] = raw["층"] / grpmax.replace(0, np.nan)
raw["층비율"] = raw["층비율"].fillna(0.5)
raw["단지평균면적"] = raw.groupby("단지키")["전용면적(㎡)"].transform("mean")
raw["재건축연한"] = (raw["건축년도"] <= 1989).astype(int)
raw["시청거리(m)"] = hav(raw["단지 위도"], raw["단지 경도"], 37.5665, 126.9780)
raw["강남역거리(m)"] = hav(raw["단지 위도"], raw["단지 경도"], 37.4979, 127.0276)
NEW = ["단지최고층", "층비율", "단지평균면적", "재건축연한", "시청거리(m)", "강남역거리(m)",
       "단지 위도", "단지 경도"]

y = np.log(raw["거래금액(만원)"] / raw["전용면적(㎡)"]).values
gu = pd.get_dummies(raw["구"], prefix="구")
FS = {"A 기존 15종 + 구": pd.concat([raw[BASE], gu], axis=1).astype(float),
      "B A + 신규 8종":   pd.concat([raw[BASE + NEW], gu], axis=1).astype(float)}
# LightGBM은 비ASCII 피처명에서 죽으므로 모델에는 f0.. 이름으로 넣고 매핑만 보관
COLMAP = {}
for k in FS:
    orig = list(FS[k].columns)
    FS[k] = FS[k].copy()
    FS[k].columns = ["f%d" % i for i in range(len(orig))]
    COLMAP[k] = dict(zip(FS[k].columns, orig))

# ---- fold seed 별 그룹 분할 ----
keys = raw["단지키"].unique()
def folds_for(seed, k=5):
    rng = np.random.default_rng(seed)
    assign = dict(zip(keys, rng.integers(0, k, len(keys))))
    f = raw["단지키"].map(assign).values
    return [(np.where(f != i)[0], np.where(f == i)[0]) for i in range(k)]

TUNED = dict(max_depth=3, learning_rate=0.0699, n_estimators=448, subsample=0.7068,
             colsample_bytree=0.6658, min_child_weight=9, reg_lambda=1.5779,
             reg_alpha=0.042, gamma=0.0864)

def build(name, seed):
    if name == "XGB 기본값":
        return xgb.XGBRegressor(objective="reg:squarederror", random_state=seed,
                                tree_method="hist", n_jobs=4)
    if name == "XGB 튜닝":
        return xgb.XGBRegressor(objective="reg:squarederror", random_state=seed,
                                tree_method="hist", n_jobs=4, **TUNED)
    if name == "LightGBM":
        return lgb.LGBMRegressor(random_state=seed, n_estimators=600, learning_rate=0.05,
                                 num_leaves=31, min_child_samples=20, subsample=0.8,
                                 subsample_freq=1, colsample_bytree=0.7, n_jobs=4, verbose=-1)
    if name == "RandomForest":
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                     random_state=seed, n_jobs=4)
    if name == "Ridge":
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))
    raise ValueError(name)

def evaluate(model_name, X, seed):
    fl = folds_for(seed)
    o = np.zeros(len(X))
    for tr, te in fl:
        m = build(model_name, seed)
        m.fit(X.iloc[tr], y[tr])
        o[te] = m.predict(X.iloc[te])
    return mean_absolute_error(np.exp(y), np.exp(o)), r2_score(y, o)

SEEDS = [0, 1, 2, 3, 4]
say("=" * 92)
say("[I] 시드 분산 · 피처 추가 · 모델 비교")
say("=" * 92)
say("2019 Q1 아파트 매매 %d행 / 단지 %d개 / fold seed 5개로 그룹 분할을 매번 새로 구성"
    % (len(raw), len(keys)))
say("종속변수 log 제곱미터당 단가, MAE 단위 만원/m2")
say("")

say("I-1. 튜닝 4.4%는 잡음 밖인가")
say("-" * 92)
say("%-14s %-18s %9s %9s %9s %9s" % ("피처셋", "모델", "MAE 평균", "MAE 표준편차", "최소", "최대"))
res = {}
for fsn, X in FS.items():
    for mn in ["XGB 기본값", "XGB 튜닝"]:
        v = np.array([evaluate(mn, X, s)[0] for s in SEEDS])
        res[(fsn, mn)] = v
        say("%-14s %-18s %9.1f %9.2f %9.1f %9.1f"
            % (fsn[:13], mn, v.mean(), v.std(ddof=1), v.min(), v.max()))
a = res[("A 기존 15종 + 구", "XGB 기본값")]; b = res[("A 기존 15종 + 구", "XGB 튜닝")]
d = a - b
say("")
say("   피처셋 A 기준 튜닝 이득 : 평균 %.2f  표준편차 %.2f  (5개 시드 전부 개선: %s)"
    % (d.mean(), d.std(ddof=1), "예" if (d > 0).all() else "아니오"))
say("   개선율 %.1f%%  /  기본값 MAE의 시드 간 표준편차 %.2f" % (d.mean() / a.mean() * 100, a.std(ddof=1)))
say("   -> 이득 %.2f 가 시드 잡음 %.2f 보다 %s"
    % (d.mean(), a.std(ddof=1), "큼" if d.mean() > a.std(ddof=1) else "작음"))

say("")
say("I-2. 신규 파생변수 8종의 효과")
say("-" * 92)
say("   추가 변수: 단지최고층, 층비율, 단지평균면적, 재건축연한, 시청거리, 강남역거리, 위도, 경도")
say("   (모두 예측 시점에 관측 가능하고 종속변수에서 파생되지 않음)")
say("")
for mn in ["XGB 기본값", "XGB 튜닝"]:
    va = res[("A 기존 15종 + 구", mn)]; vb = res[("B A + 신규 8종", mn)]
    say("   %-12s A %.1f (±%.2f)  ->  B %.1f (±%.2f)   개선 %.1f%%   5시드 전부 개선: %s"
        % (mn, va.mean(), va.std(ddof=1), vb.mean(), vb.std(ddof=1),
           (va.mean() - vb.mean()) / va.mean() * 100, "예" if ((va - vb) > 0).all() else "아니오"))

say("")
say("I-3. 모델 비교 (피처셋 B, 시드 5개)")
say("-" * 92)
say("%-16s %10s %10s %10s %10s" % ("모델", "MAE 평균", "표준편차", "R2 평균", "R2 표준편차"))
XB = FS["B A + 신규 8종"]
rank = []
for mn in ["XGB 튜닝", "LightGBM", "RandomForest", "Ridge", "XGB 기본값"]:
    try:
        vs = [evaluate(mn, XB, s) for s in SEEDS]
    except Exception as e:
        say("%-16s 실행 실패: %s" % (mn, str(e)[:50])); continue
    mae = np.array([v[0] for v in vs]); r2 = np.array([v[1] for v in vs])
    rank.append((mae.mean(), mn))
    say("%-16s %10.1f %10.2f %10.4f %10.4f"
        % (mn, mae.mean(), mae.std(ddof=1), r2.mean(), r2.std(ddof=1)))
rank.sort()
say("")
say("   최고 %s (%.1f) / 차순 %s (%.1f) / 격차 %.1f"
    % (rank[0][1], rank[0][0], rank[1][1], rank[1][0], rank[1][0] - rank[0][0]))
say("   CatBoost는 이 환경에 설치되어 있지 않아 비교에서 제외")

say("")
say("I-4. 신규 변수의 중요도 (XGB 튜닝, 피처셋 B, 시드 0)")
say("-" * 92)
m = build("XGB 튜닝", 0); m.fit(XB, y)
names = [COLMAP["B A + 신규 8종"][c] for c in XB.columns]
s = pd.Series(m.feature_importances_, index=names)
loc = s[[c for c in s.index if c.startswith("구_")]].sum()
s = s[[c for c in s.index if not c.startswith("구_")]]
s["[구 더미 합계]"] = loc
for k, v in s.sort_values(ascending=False).head(12).items():
    mark = "  <= 신규" if k in NEW else ""
    say("   %-18s %.4f%s" % (k, v, mark))

P = RESULTS_DIR
open(P + r"\out_part9.txt", "w", encoding="utf-8").write("\n".join(OUT))
