# -*- coding: utf-8 -*-
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import DATA_DIR, RESULTS_DIR
"""1순위 · 2순위 작업
  1) 역세권 500m 임계와 규모별 U자가 20칸에서도 성립하는가
  2) 파생변수 8종의 효과가 20칸에서도 나오는가
"""
import numpy as np, pandas as pd, xgboost as xgb, gc
from sklearn.metrics import mean_absolute_error

OUT = []
def say(s=""):
    OUT.append(str(s)); print(s)

B = DATA_DIR
INF = ["편의점(개)", "약국(개)", "병원(개)", "공공도서관(개)", "경찰서(개)", "소방서(개)",
       "대형마트(개)", "초등학교 거리(m)", "중학교 거리(m)", "고등학교 거리(m)"]
KEEP = ["시군구", "단지명", "전용면적(㎡)", "층", "건축년도", "년도", "분기", "거리(m)",
        "최단거리(m)", "거래금액(만원)", "보증금(만원)", "월세금(만원)", "전월세구분",
        "단지 위도", "단지 경도"] + INF
TP = [(2019, 1), (2020, 2), (2021, 4), (2022, 4)]
TUNED = dict(max_depth=3, learning_rate=0.0699, n_estimators=448, subsample=0.7068,
             colsample_bytree=0.6658, min_child_weight=9, reg_lambda=1.5779,
             reg_alpha=0.042, gamma=0.0864)

def mk():
    return xgb.XGBRegressor(objective="reg:squarederror", random_state=42,
                            tree_method="hist", n_jobs=3, **TUNED)

def num(s):
    if s.dtype == object:
        s = s.astype(str).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(s, errors="coerce")

def hav(la1, lo1, la2, lo2):
    R = 6371.0
    p1, p2 = np.radians(la1), np.radians(la2)
    dp, dl = np.radians(la2 - la1), np.radians(lo2 - lo1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)) * 1000

def load(f):
    d = pd.read_csv(B + "\\" + f, encoding="CP949", low_memory=False,
                    usecols=lambda c: c in KEEP)
    d = d.rename(columns={"거리(m)": "지하철역까지 거리(m)", "최단거리(m)": "정류소까지 거리(m)"})
    sp = d["시군구"].str.split()
    d["구"] = sp.str[1]; d["동"] = sp.str[2]
    d["단지키"] = d["구"] + "_" + d["동"] + "_" + d["단지명"].astype(str)
    for c in (["전용면적(㎡)", "층", "건축년도", "년도", "분기", "거래금액(만원)",
               "보증금(만원)", "월세금(만원)", "지하철역까지 거리(m)", "정류소까지 거리(m)",
               "단지 위도", "단지 경도"] + INF):
        if c in d.columns:
            d[c] = num(d[c])
    return d

SEG = [("아파트 전세", "아파트_전월세.csv", "전세", "보증금(만원)"),
       ("아파트 월세", "아파트_전월세.csv", "월세", "보증금(만원)"),
       ("연립 매매", "연립다세대_매매.csv", None, "거래금액(만원)"),
       ("연립 전세", "연립다세대_전월세.csv", "전세", "보증금(만원)"),
       ("연립 월세", "연립다세대_전월세.csv", "월세", "보증금(만원)")]

GRID = [100, 300, 500, 700, 1000, 1500, 2000]
cache = {}
FEAT, SUB, SIZE = [], [], []

for name, f, gubun, tgt in SEG:
    if f not in cache:
        cache[f] = load(f)
    d0 = cache[f]
    if gubun:
        d0 = d0[d0["전월세구분"] == gubun]
    BASE = ["전용면적(㎡)", "층", "건축년도"] + INF + ["지하철역까지 거리(m)", "정류소까지 거리(m)"]
    if gubun == "월세":
        BASE = BASE + ["월세금(만원)"]
    d0 = d0.dropna(subset=BASE + [tgt, "단지 위도", "단지 경도"])
    d0 = d0[(d0[tgt] > 0) & (d0["전용면적(㎡)"] > 0)]

    for yy, qq in TP:
        s = d0[(d0["년도"] == yy) & (d0["분기"] == qq)].copy().reset_index(drop=True)
        if len(s) < 300:
            continue
        tag = "%dQ%d" % (yy, qq)
        gm = s.groupby("단지키")["층"].transform("max")
        s["단지최고층"] = gm
        s["층비율"] = (s["층"] / gm.replace(0, np.nan)).fillna(0.5)
        s["단지평균면적"] = s.groupby("단지키")["전용면적(㎡)"].transform("mean")
        s["재건축연한"] = (s["건축년도"] <= 1989).astype(int)
        s["시청거리(m)"] = hav(s["단지 위도"], s["단지 경도"], 37.5665, 126.9780)
        s["강남역거리(m)"] = hav(s["단지 위도"], s["단지 경도"], 37.4979, 127.0276)
        NEW = ["단지최고층", "층비율", "단지평균면적", "재건축연한", "시청거리(m)",
               "강남역거리(m)", "단지 위도", "단지 경도"]

        unit = (s[tgt] / s["전용면적(㎡)"]).values
        yl = np.log(unit)
        gu = pd.get_dummies(s["구"], prefix="구")
        XA = pd.concat([s[BASE], gu], axis=1).astype(float)
        XB = pd.concat([s[BASE + NEW], gu], axis=1).astype(float)
        XA.columns = ["f%d" % i for i in range(XA.shape[1])]
        colsB = list(XB.columns)
        XB.columns = ["g%d" % i for i in range(XB.shape[1])]

        rng = np.random.default_rng(0)
        keys = s["단지키"].unique()
        assign = dict(zip(keys, rng.integers(0, 5, len(keys))))
        fid = s["단지키"].map(assign).values
        folds = [(np.where(fid != i)[0], np.where(fid == i)[0]) for i in range(5)]

        def oof(X):
            o = np.zeros(len(X))
            for tr, te in folds:
                o[te] = mk().fit(X.iloc[tr], yl[tr]).predict(X.iloc[te])
            return mean_absolute_error(np.exp(yl), np.exp(o))
        ma, mb = oof(XA), oof(XB)
        FEAT.append((name, tag, len(s), ma, mb, (ma - mb) / ma * 100))

        # 역세권 PDP (피처셋 A, 전체 적합)
        mf = mk().fit(XA, yl)
        col = "f%d" % BASE.index("지하철역까지 거리(m)")
        pv = []
        for v in GRID:
            Z = XA.copy(); Z[col] = v
            pv.append(float(np.exp(mf.predict(Z)).mean()))
        near = pv[0] - pv[2]           # 100m -> 500m 하락
        far = pv[2] - pv[4]            # 500m -> 1000m 하락
        SUB.append((name, tag, pv, near, far, far / near if near > 0 else np.inf))

        # 규모별 단가 (모델 없이 실제 값)
        s["규모"] = pd.cut(s["전용면적(㎡)"], [0, 40, 60, 85, 135, 1e9],
                         labels=["소형", "중소형", "중형", "중대형", "대형"])
        med = s.assign(단가=unit).groupby("규모", observed=True)["단가"].median()
        med = med.reindex(["소형", "중소형", "중형", "중대형", "대형"])
        vals = med.values
        ok = np.isfinite(vals)
        isU = bool(ok.all() and np.nanargmin(vals) in (1, 2, 3))
        SIZE.append((name, tag, vals, isU))
        del XA, XB; gc.collect()
    gc.collect()

say("=" * 96)
say("[J] 1순위 · 2순위 : 한 칸 근거 주장 3개를 20칸으로 확장")
say("=" * 96)

say("")
say("J-1. 파생변수 8종의 효과가 20칸에서도 나오는가   MAE 만원/m2")
say("-" * 96)
say("%-10s %-8s %8s %10s %10s %9s" % ("세그먼트", "시점", "표본", "A 기존", "B +8종", "개선"))
for r in FEAT:
    say("%-10s %-8s %8d %10.1f %10.1f %8.1f%%" % r)
imp = np.array([r[5] for r in FEAT])
say("-" * 96)
say("   20칸 중 개선된 칸 : %d / %d" % ((imp > 0).sum(), len(imp)))
say("   개선율 평균 %.1f%% / 최소 %.1f%% / 최대 %.1f%%" % (imp.mean(), imp.min(), imp.max()))
say("   (2019Q1 아파트 매매 단일 칸에서 얻었던 값은 16.3%)")

say("")
say("J-2. 역세권 500m 임계가 20칸에서도 보이는가")
say("-" * 96)
say("   판정 기준: 500m~1000m 구간 하락폭이 100m~500m 구간 하락폭보다 큰가")
say("%-10s %-8s %11s %11s %9s  %s" % ("세그먼트", "시점", "100→500", "500→1000", "배율", "임계"))
ratios = []
for name, tag, pv, near, far, ratio in SUB:
    ratios.append(ratio)
    say("%-10s %-8s %11.1f %11.1f %9.2f  %s"
        % (name, tag, near, far, ratio, "O" if ratio > 1 else "X"))
ra = np.array(ratios)
say("-" * 96)
say("   500m 임계가 보이는 칸 : %d / %d" % ((ra > 1).sum(), len(ra)))
say("   배율 중앙값 %.2f" % np.median(ra))
say("   (2019Q1 아파트 매매에서는 63 대 343 으로 배율 5.4 였다)")

say("")
say("J-3. 규모별 단가 U자가 20칸에서도 나오는가")
say("-" * 96)
say("   판정 기준: 소형~대형 5구간 중 최저값이 가운데 세 구간에 있는가")
say("%-10s %-8s %8s %8s %8s %8s %8s  %s"
    % ("세그먼트", "시점", "소형", "중소형", "중형", "중대형", "대형", "U자"))
u = 0
for name, tag, vals, isU in SIZE:
    if isU:
        u += 1
    say("%-10s %-8s %8.0f %8.0f %8.0f %8.0f %8.0f  %s"
        % (name, tag, *[v if np.isfinite(v) else -1 for v in vals], "O" if isU else "X"))
say("-" * 96)
say("   U자인 칸 : %d / %d" % (u, len(SIZE)))
say("   (2019Q1 아파트 매매에서는 2802 / 2355 / 2274 / 2191 / 3342 로 U자였다)")

P = RESULTS_DIR
open(P + r"\out_part12.txt", "w", encoding="utf-8").write("\n".join(OUT))
pd.DataFrame(FEAT, columns=["세그먼트", "시점", "n", "MAE_A", "MAE_B", "개선율"]).to_csv(
    P + r"\j_feature.csv", index=False, encoding="utf-8-sig")
