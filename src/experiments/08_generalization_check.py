# -*- coding: utf-8 -*-
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import DATA_DIR, RESULTS_DIR
import numpy as np, pandas as pd, xgboost as xgb, shap, gc
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import spearmanr

OUT = []
def say(s=""):
    OUT.append(str(s)); print(s)

B = DATA_DIR
INF = ["편의점(개)","약국(개)","병원(개)","공공도서관(개)","경찰서(개)","소방서(개)","대형마트(개)",
       "초등학교 거리(m)","중학교 거리(m)","고등학교 거리(m)"]
KEEP = ["시군구","단지명","전용면적(㎡)","층","건축년도","년도","분기","거리(m)","최단거리(m)",
        "거래금액(만원)","보증금(만원)","월세금(만원)","전월세구분"] + INF
TP = [(2019,1),(2020,2),(2021,4),(2022,4)]

def mk():
    return xgb.XGBRegressor(objective="reg:squarederror", random_state=42,
                            tree_method="hist", n_jobs=4)

def num(s):
    if s.dtype == object:
        s = s.astype(str).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(s, errors="coerce")

def load(f):
    d = pd.read_csv(B + "\\" + f, encoding="CP949", low_memory=False,
                    usecols=lambda c: c in KEEP)
    d = d.rename(columns={"거리(m)":"지하철역까지 거리(m)", "최단거리(m)":"정류소까지 거리(m)"})
    sp = d["시군구"].str.split()
    d["구"] = sp.str[1]; d["동"] = sp.str[2]
    d["단지키"] = d["구"] + "_" + d["동"] + "_" + d["단지명"].astype(str)
    for c in (["전용면적(㎡)","층","건축년도","년도","분기","거래금액(만원)","보증금(만원)",
               "월세금(만원)","지하철역까지 거리(m)","정류소까지 거리(m)"] + INF):
        if c in d.columns:
            d[c] = num(d[c])
    return d

SEG = [("아파트 전세","아파트_전월세.csv","전세","보증금(만원)"),
       ("아파트 월세","아파트_전월세.csv","월세","보증금(만원)"),
       ("연립 매매","연립다세대_매매.csv",None,"거래금액(만원)"),
       ("연립 전세","연립다세대_전월세.csv","전세","보증금(만원)"),
       ("연립 월세","연립다세대_전월세.csv","월세","보증금(만원)")]

cache = {}
LADDER = []; SHAPD = []; PDPD = []
SHAPCOLS = ["편의점(개)","병원(개)","지하철역까지 거리(m)","건축년도","전용면적(㎡)"]

for name, f, gubun, tgt in SEG:
    if f not in cache:
        cache[f] = load(f)
    d = cache[f]
    if gubun:
        d = d[d["전월세구분"] == gubun]
    F = ["전용면적(㎡)","층","건축년도"] + INF + ["지하철역까지 거리(m)","정류소까지 거리(m)"]
    if gubun == "월세":
        F = F + ["월세금(만원)"]
    d0 = d.dropna(subset=F + [tgt])
    d0 = d0[(d0[tgt] > 0) & (d0["전용면적(㎡)"] > 0)]

    for y, q in TP:
        s = d0[(d0["년도"] == y) & (d0["분기"] == q)].reset_index(drop=True)
        if len(s) < 300:
            continue
        unit = (s[tgt] / s["전용면적(㎡)"]).values
        yl = np.log(unit)
        grp = s["단지키"].values
        X3 = s[F]
        X4 = pd.concat([s[F], pd.get_dummies(s["구"], prefix="구")], axis=1).astype(float)
        folds = list(GroupKFold(n_splits=5).split(X3, yl, grp))

        def oof(X):
            pred = np.zeros(len(X))
            for tr, te in folds:
                pred[te] = mk().fit(X.iloc[tr], yl[tr]).predict(X.iloc[te])
            return pred

        # B1 baseline : 동별 중앙값
        ob = np.zeros(len(s)); dong = s["동"].values
        for tr, te in folds:
            med = pd.Series(unit[tr]).groupby(pd.Series(dong[tr])).median()
            gl = np.median(unit[tr])
            ob[te] = np.log(pd.Series(dong[te]).map(med).fillna(gl).values)
        o3 = oof(X3); o4 = oof(X4)

        def mae(pred):
            return mean_absolute_error(np.exp(yl), np.exp(pred))

        LADDER.append((name, "%dQ%d" % (y, q), len(s),
                       mae(ob), mae(o3), mae(o4),
                       r2_score(yl, ob), r2_score(yl, o3), r2_score(yl, o4)))

        # SHAP : 첫 폴드 단일 적합
        tr, te = folds[0]
        mm = mk().fit(X4.iloc[tr], yl[tr])
        sv = pd.DataFrame(shap.TreeExplainer(mm).shap_values(X4.iloc[te]), columns=X4.columns)
        row = {"세그먼트": name, "시점": "%dQ%d" % (y, q)}
        for c in SHAPCOLS:
            row[c] = round(float(spearmanr(X4.iloc[te][c], sv[c]).statistic), 2)
        SHAPD.append(row)

        # PDP 건축년도 : 전체 적합
        mf = mk().fit(X4, yl)
        grid = [1975, 1985, 1995, 2005, 2015]
        pv = []
        for v in grid:
            Z = X4.copy(); Z["건축년도"] = v
            pv.append(float(np.exp(mf.predict(Z)).mean() * 3.3058))
        PDPD.append((name, "%dQ%d" % (y, q), pv, pv[1] > pv[2]))
        del X3, X4; gc.collect()
    gc.collect()

say("=" * 96)
say("[H] 핵심 주장 3개의 일반화 검증 (5유형 x 4시점 = 20칸)")
say("=" * 96)
say("")
say("H-1. 인프라 15종이 동별 중앙값보다 나은가   MAE 만원/m2, 낮을수록 좋음")
say("-" * 96)
say("%-10s %-8s %8s | %9s %9s %9s | %s"
    % ("세그먼트","시점","표본","B1 동중앙","M3 15종","M4a +구","M3가 B1보다"))
win = 0
for r in LADDER:
    v = "나음" if r[4] < r[3] else "못함"
    if r[4] < r[3]:
        win += 1
    say("%-10s %-8s %8d | %9.1f %9.1f %9.1f | %s" % (r[0], r[1], r[2], r[3], r[4], r[5], v))
say("-" * 96)
say("   M3(인프라 15종)가 B1(동별 중앙값)보다 나은 칸 : %d / %d" % (win, len(LADDER)))
w4 = sum(1 for r in LADDER if r[5] < r[3])
say("   M4a(+구 더미)가 B1보다 나은 칸               : %d / %d" % (w4, len(LADDER)))
say("   구 더미 추가의 평균 오차 감소                : %.1f%%"
    % np.mean([(r[4] - r[5]) / r[4] * 100 for r in LADDER]))

say("")
say("H-2. SHAP 방향이 세그먼트마다 같은가   변수값과 SHAP값의 스피어만 상관")
say("-" * 96)
T = pd.DataFrame(SHAPD)
say(T.to_string(index=False))
say("")
for c in SHAPCOLS:
    v = T[c].values
    sign = "일관 음(-)" if (v < 0).all() else ("일관 양(+)" if (v > 0).all() else "혼재")
    say("   %-18s 음수 %2d / 양수 %2d  ->  %s" % (c, (v < 0).sum(), (v > 0).sum(), sign))

say("")
say("H-3. 건축년도 U자   1985년 예측 평당가가 1995년보다 높은가")
say("-" * 96)
say("%-10s %-8s %9s %9s %9s %9s %9s  %s"
    % ("세그먼트","시점","1975","1985","1995","2005","2015","U자"))
u = 0
for name, tp, pv, isU in PDPD:
    if isU:
        u += 1
    say("%-10s %-8s %9.0f %9.0f %9.0f %9.0f %9.0f  %s"
        % (name, tp, pv[0], pv[1], pv[2], pv[3], pv[4], "O" if isU else "X"))
say("-" * 96)
say("   1985 > 1995 인 칸 : %d / %d" % (u, len(PDPD)))

OUTP = RESULTS_DIR
open(OUTP + r"\out_part8.txt", "w", encoding="utf-8").write("\n".join(OUT))
pd.DataFrame(LADDER, columns=["세그먼트","시점","n","MAE_B1","MAE_M3","MAE_M4a",
                              "R2_B1","R2_M3","R2_M4a"]).to_csv(
    OUTP + r"\h_ladder.csv", index=False, encoding="utf-8-sig")
T.to_csv(OUTP + r"\h_shap.csv", index=False, encoding="utf-8-sig")
