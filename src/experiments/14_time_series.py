# -*- coding: utf-8 -*-
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import DATA_DIR, RESULTS_DIR
"""시계열 구조 도입
  A) 시간 기반 분할(forward chaining) — 과거로 학습해 다음 분기를 맞춘다
  B) 직전 분기 파생변수 — 동별 lag 단가, 이동평균, 모멘텀, 거래량
  C) 20분기 전체 중요도 추이 — 슬라이드의 '시장 국면' 서사를 제대로 검증

누수 방지: t분기 예측에 쓰는 lag 변수는 t보다 앞선 분기 데이터로만 만든다.
"""
import sys
import numpy as np, pandas as pd, xgboost as xgb, gc
from sklearn.metrics import mean_absolute_error, r2_score

SEGNAME = sys.argv[1]           # "아파트 전세" | "연립 매매"
OUT = []
def say(s=""):
    OUT.append(str(s)); print(s)

B = DATA_DIR
INF = ["편의점(개)", "약국(개)", "병원(개)", "공공도서관(개)", "경찰서(개)", "소방서(개)",
       "대형마트(개)", "초등학교 거리(m)", "중학교 거리(m)", "고등학교 거리(m)"]
KEEP = ["시군구", "단지명", "전용면적(㎡)", "층", "건축년도", "년도", "분기", "거리(m)",
        "최단거리(m)", "거래금액(만원)", "보증금(만원)", "전월세구분"] + INF
CFG = {"아파트 전세": ("아파트_전월세.csv", "전세", "보증금(만원)"),
       "연립 매매": ("연립다세대_매매.csv", None, "거래금액(만원)")}
FILE, GUBUN, TGT = CFG[SEGNAME]
TUNED = dict(max_depth=6, learning_rate=0.07, n_estimators=400, subsample=0.8,
             colsample_bytree=0.8, min_child_weight=10, reg_lambda=2.0)

def mk():
    return xgb.XGBRegressor(objective="reg:squarederror", random_state=42,
                            tree_method="hist", n_jobs=4, **TUNED)

def num(s):
    if s.dtype == object:
        s = s.astype(str).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(s, errors="coerce")

d = pd.read_csv(B + "\\" + FILE, encoding="CP949", low_memory=False,
                usecols=lambda c: c in KEEP)
d = d.rename(columns={"거리(m)": "지하철역까지 거리(m)", "최단거리(m)": "정류소까지 거리(m)"})
sp = d["시군구"].str.split()
d["구"] = sp.str[1]; d["동"] = sp.str[2]
if GUBUN:
    d = d[d["전월세구분"] == GUBUN]
BASE = ["전용면적(㎡)", "층", "건축년도"] + INF + ["지하철역까지 거리(m)", "정류소까지 거리(m)"]
for c in BASE + [TGT, "년도", "분기"]:
    d[c] = num(d[c])
d = d.dropna(subset=BASE + [TGT, "동"])
d = d[(d[TGT] > 0) & (d["전용면적(㎡)"] > 0)].reset_index(drop=True)
d["t"] = (d["년도"] - 2019) * 4 + d["분기"]          # 1 .. 20
d = d[(d["t"] >= 1) & (d["t"] <= 20)].reset_index(drop=True)
d["단가"] = d[TGT] / d["전용면적(㎡)"]
d["구동"] = d["구"] + "_" + d["동"]

say("=" * 92)
say("[L] 시계열 구조 도입 · %s" % SEGNAME)
say("=" * 92)
say("표본 %d행 / 분기 %d개 / 동 %d개" % (len(d), d["t"].nunique(), d["구동"].nunique()))

# ---------- B) 직전 분기 파생변수 ----------
# 동×분기 집계표를 만들고 shift 해서 붙인다. t 시점에는 t-1 이전 정보만 들어간다.
agg = d.groupby(["구동", "t"])["단가"].agg(["median", "size"]).reset_index()
agg = agg.rename(columns={"median": "m", "size": "n"})
full = pd.MultiIndex.from_product([agg["구동"].unique(), range(1, 21)],
                                  names=["구동", "t"]).to_frame(index=False)
agg = full.merge(agg, on=["구동", "t"], how="left")
agg = agg.sort_values(["구동", "t"])
g = agg.groupby("구동", sort=False)
agg["lag1"] = g["m"].shift(1)
agg["lag_ma4"] = g["m"].shift(1).rolling(4, min_periods=1).median().reset_index(level=0, drop=True)
agg["lag_vol"] = g["n"].shift(1)
agg["모멘텀"] = agg["lag1"] / agg["lag_ma4"] - 1

gagg = d.groupby(["구", "t"])["단가"].median().reset_index().rename(columns={"단가": "gm"})
gfull = pd.MultiIndex.from_product([gagg["구"].unique(), range(1, 21)],
                                   names=["구", "t"]).to_frame(index=False)
gagg = gfull.merge(gagg, on=["구", "t"], how="left").sort_values(["구", "t"])
gagg["구_lag1"] = gagg.groupby("구", sort=False)["gm"].shift(1)
gagg = gagg.drop(columns=["gm"])

d = d.merge(agg[["구동", "t", "lag1", "lag_ma4", "lag_vol", "모멘텀"]], on=["구동", "t"], how="left")
d = d.merge(gagg, on=["구", "t"], how="left")
d["lag1"] = d["lag1"].fillna(d["구_lag1"])
d["lag_ma4"] = d["lag_ma4"].fillna(d["구_lag1"])
LAG = ["lag1", "lag_ma4", "lag_vol", "모멘텀", "구_lag1"]

y = np.log(d["단가"].values)
gu = pd.get_dummies(d["구"], prefix="구")
XA = pd.concat([d[BASE], gu], axis=1).astype(float)
XB = pd.concat([d[BASE + LAG], gu], axis=1).astype(float)
nameA, nameB = list(XA.columns), list(XB.columns)
XA.columns = ["a%d" % i for i in range(XA.shape[1])]
XB.columns = ["b%d" % i for i in range(XB.shape[1])]
tv = d["t"].values

# ---------- A) forward chaining ----------
say("")
say("A+B. 시간 기반 분할 : 1..t-1 분기로 학습해 t 분기를 예측")
say("-" * 92)
say("%-8s %8s %10s %10s %9s" % ("평가분기", "표본", "변수만", "+시계열", "개선"))
rows = []
for t in range(5, 21):
    tr = np.where(tv < t)[0]; te = np.where(tv == t)[0]
    if len(te) < 200:
        continue
    ma = mean_absolute_error(np.exp(y[te]),
                             np.exp(mk().fit(XA.iloc[tr], y[tr]).predict(XA.iloc[te])))
    mb = mean_absolute_error(np.exp(y[te]),
                             np.exp(mk().fit(XB.iloc[tr], y[tr]).predict(XB.iloc[te])))
    rows.append((t, len(te), ma, mb, (ma - mb) / ma * 100))
    say("%-8s %8d %10.1f %10.1f %8.1f%%"
        % ("%dQ%d" % (2019 + (t - 1) // 4, (t - 1) % 4 + 1), len(te), ma, mb, rows[-1][4]))
imp = np.array([r[4] for r in rows])
mas = np.array([r[2] for r in rows]); mbs = np.array([r[3] for r in rows])
say("-" * 92)
say("   평가 분기 %d개 | 변수만 평균 %.1f | +시계열 평균 %.1f | 개선 평균 %.1f%%"
    % (len(rows), mas.mean(), mbs.mean(), imp.mean()))
say("   개선된 분기 %d / %d" % ((imp > 0).sum(), len(imp)))

# 같은 분기 안에서 나눴을 때와 대조
say("")
say("A-2. 같은 분기 안에서 나눴을 때와 대조 (누수 있는 설계)")
say("-" * 92)
say("%-8s %10s %10s %10s" % ("평가분기", "분기내분할", "시간분할", "차이"))
for t in [8, 12, 16, 20]:
    idx = np.where(tv == t)[0]
    if len(idx) < 400:
        continue
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(idx)); cut = int(len(idx) * 0.8)
    itr, ite = idx[perm[:cut]], idx[perm[cut:]]
    within = mean_absolute_error(np.exp(y[ite]),
                                 np.exp(mk().fit(XB.iloc[itr], y[itr]).predict(XB.iloc[ite])))
    fwd = [r[3] for r in rows if r[0] == t]
    if fwd:
        say("%-8s %10.1f %10.1f %10.1f"
            % ("%dQ%d" % (2019 + (t - 1) // 4, (t - 1) % 4 + 1), within, fwd[0], fwd[0] - within))

# ---------- C) 20분기 중요도 추이 ----------
say("")
say("C. 20분기 전체에서 전용면적 중요도 추이 (분기별 개별 학습, 거래금액 종속변수)")
say("-" * 92)
tr_imp = []
for t in range(1, 21):
    idx = np.where(tv == t)[0]
    if len(idx) < 300:
        continue
    m = mk().fit(XA.iloc[idx], np.log(d[TGT].values[idx]))
    s = pd.Series(m.feature_importances_, index=nameA)
    tr_imp.append((t, len(idx), float(s["전용면적(㎡)"]), s.idxmax()))
say("%-8s %8s %10s  %s" % ("분기", "표본", "전용면적", "1순위"))
for t, n, v, top in tr_imp:
    say("%-8s %8d %10.3f  %s" % ("%dQ%d" % (2019 + (t - 1) // 4, (t - 1) % 4 + 1), n, v, top))
vv = np.array([r[2] for r in tr_imp])
say("-" * 92)
say("   20분기 전용면적 중요도  최소 %.3f / 최대 %.3f / 표준편차 %.3f"
    % (vv.min(), vv.max(), vv.std(ddof=1)))
say("   1순위가 전용면적인 분기 %d / %d"
    % (sum(1 for r in tr_imp if r[3] == "전용면적(㎡)"), len(tr_imp)))

P = RESULTS_DIR
open(P + "\\out_part14_%s.txt" % SEGNAME.replace(" ", ""), "w",
     encoding="utf-8").write("\n".join(OUT))
