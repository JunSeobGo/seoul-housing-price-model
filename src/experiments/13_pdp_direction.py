# -*- coding: utf-8 -*-
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import DATA_DIR, RESULTS_DIR
"""3순위 · SHAP 상관 대신 부분의존도로 방향을 다시 뽑는다.

기존 방식: spearman(변수값, SHAP값) 한 숫자
문제: 관계가 단조가 아니면 이 숫자가 의미를 잃는다. 20칸에서 편의점이
      '혼재'로 나온 게 실제 부호가 갈려서인지 방법 탓인지 구분이 안 된다.

새 방식: 변수의 10분위 격자에서 부분의존도를 계산해
  effect   = (마지막 - 처음) / 전체 평균, 즉 상대 효과 크기
  mono     = spearman(격자, 부분의존도), 곡선이 단조인가
  shape    = 최저점 위치로 U자 / 역U자 / 단조 판정
"""
import numpy as np, pandas as pd, xgboost as xgb, gc
from scipy.stats import spearmanr

OUT = []
def say(s=""):
    OUT.append(str(s)); print(s)

B = DATA_DIR
INF = ["편의점(개)", "약국(개)", "병원(개)", "공공도서관(개)", "경찰서(개)", "소방서(개)",
       "대형마트(개)", "초등학교 거리(m)", "중학교 거리(m)", "고등학교 거리(m)"]
KEEP = ["시군구", "단지명", "전용면적(㎡)", "층", "건축년도", "년도", "분기", "거리(m)",
        "최단거리(m)", "거래금액(만원)", "보증금(만원)", "월세금(만원)", "전월세구분"] + INF
TP = [(2019, 1), (2020, 2), (2021, 4), (2022, 4)]
TUNED = dict(max_depth=3, learning_rate=0.0699, n_estimators=448, subsample=0.7068,
             colsample_bytree=0.6658, min_child_weight=9, reg_lambda=1.5779,
             reg_alpha=0.042, gamma=0.0864)
WATCH = ["편의점(개)", "병원(개)", "지하철역까지 거리(m)", "건축년도", "전용면적(㎡)"]

def mk():
    return xgb.XGBRegressor(objective="reg:squarederror", random_state=42,
                            tree_method="hist", n_jobs=3, **TUNED)

def num(s):
    if s.dtype == object:
        s = s.astype(str).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(s, errors="coerce")

def load(f):
    d = pd.read_csv(B + "\\" + f, encoding="CP949", low_memory=False,
                    usecols=lambda c: c in KEEP)
    d = d.rename(columns={"거리(m)": "지하철역까지 거리(m)", "최단거리(m)": "정류소까지 거리(m)"})
    sp = d["시군구"].str.split()
    d["구"] = sp.str[1]; d["동"] = sp.str[2]
    for c in (["전용면적(㎡)", "층", "건축년도", "년도", "분기", "거래금액(만원)",
               "보증금(만원)", "월세금(만원)", "지하철역까지 거리(m)",
               "정류소까지 거리(m)"] + INF):
        if c in d.columns:
            d[c] = num(d[c])
    return d

SEG = [("아파트 전세", "아파트_전월세.csv", "전세", "보증금(만원)"),
       ("아파트 월세", "아파트_전월세.csv", "월세", "보증금(만원)"),
       ("연립 매매", "연립다세대_매매.csv", None, "거래금액(만원)"),
       ("연립 전세", "연립다세대_전월세.csv", "전세", "보증금(만원)"),
       ("연립 월세", "연립다세대_전월세.csv", "월세", "보증금(만원)")]

cache = {}
ROWS = []
for name, f, gubun, tgt in SEG:
    if f not in cache:
        cache[f] = load(f)
    d0 = cache[f]
    if gubun:
        d0 = d0[d0["전월세구분"] == gubun]
    BASE = ["전용면적(㎡)", "층", "건축년도"] + INF + ["지하철역까지 거리(m)", "정류소까지 거리(m)"]
    if gubun == "월세":
        BASE = BASE + ["월세금(만원)"]
    d0 = d0.dropna(subset=BASE + [tgt])
    d0 = d0[(d0[tgt] > 0) & (d0["전용면적(㎡)"] > 0)]
    for yy, qq in TP:
        s = d0[(d0["년도"] == yy) & (d0["분기"] == qq)].copy().reset_index(drop=True)
        if len(s) < 300:
            continue
        tag = "%dQ%d" % (yy, qq)
        yl = np.log((s[tgt] / s["전용면적(㎡)"]).values)
        X = pd.concat([s[BASE], pd.get_dummies(s["구"], prefix="구")], axis=1).astype(float)
        names = list(X.columns)
        X.columns = ["f%d" % i for i in range(X.shape[1])]
        m = mk().fit(X, yl)
        for w in WATCH:
            col = "f%d" % names.index(w)
            grid = np.unique(np.quantile(X[col], np.linspace(0.05, 0.95, 10)))
            if len(grid) < 4:
                continue
            pv = []
            for v in grid:
                Z = X.copy(); Z[col] = v
                pv.append(float(np.exp(m.predict(Z)).mean()))
            pv = np.array(pv)
            effect = (pv[-1] - pv[0]) / pv.mean() * 100
            mono = float(spearmanr(grid, pv).statistic)
            lo = int(np.argmin(pv)); hi = int(np.argmax(pv)); n = len(pv)
            if abs(mono) > 0.9:
                shape = "단조증가" if mono > 0 else "단조감소"
            elif 0 < lo < n - 1:
                shape = "U자"
            elif 0 < hi < n - 1:
                shape = "역U자"
            else:
                shape = "비단조"
            ROWS.append((name, tag, w, effect, mono, shape))
        del X; gc.collect()
    gc.collect()

R = pd.DataFrame(ROWS, columns=["세그먼트", "시점", "변수", "효과%", "단조성", "형태"])

say("=" * 94)
say("[K] 3순위 : SHAP 상관 대신 부분의존도로 방향 재산출 (20칸)")
say("=" * 94)
say("효과%는 변수의 5분위에서 95분위로 갈 때 예측 단가의 상대 변화")
say("단조성은 격자와 부분의존도의 스피어만 상관, 1에 가까우면 단조증가")
say("")

for w in WATCH:
    sub = R[R["변수"] == w]
    pos = (sub["효과%"] > 0).sum(); neg = (sub["효과%"] < 0).sum()
    say("── %s" % w)
    say("   %-10s %-8s %9s %9s  %s" % ("세그먼트", "시점", "효과%", "단조성", "형태"))
    for _, r in sub.iterrows():
        say("   %-10s %-8s %9.1f %9.2f  %s"
            % (r["세그먼트"], r["시점"], r["효과%"], r["단조성"], r["형태"]))
    shapes = sub["형태"].value_counts().to_dict()
    say("   부호  양 %d / 음 %d      형태  %s"
        % (pos, neg, ", ".join("%s %d" % (k, v) for k, v in shapes.items())))
    say("")

say("=" * 94)
say("[K-2] 기존 SHAP 상관 판정과 대조")
say("=" * 94)
prev = {"편의점(개)": "혼재 (음 10 / 양 10)", "병원(개)": "일관 양 (20/0)",
        "지하철역까지 거리(m)": "일관 음 (0/20)", "건축년도": "일관 양 (20/0)",
        "전용면적(㎡)": "혼재 (음 11 / 양 9)"}
say("%-20s %-26s %-26s %s" % ("변수", "SHAP 상관 판정", "부분의존도 판정", "단조인 칸"))
for w in WATCH:
    sub = R[R["변수"] == w]
    pos = (sub["효과%"] > 0).sum(); neg = (sub["효과%"] < 0).sum()
    newv = "일관 양 (%d/%d)" % (pos, len(sub)) if neg == 0 else (
        "일관 음 (%d/%d)" % (neg, len(sub)) if pos == 0 else "혼재 (음 %d / 양 %d)" % (neg, pos))
    monoc = int(sub["형태"].isin(["단조증가", "단조감소"]).sum())
    say("%-20s %-26s %-26s %d / %d" % (w, prev[w], newv, monoc, len(sub)))

P = RESULTS_DIR
open(P + r"\out_part13.txt", "w", encoding="utf-8").write("\n".join(OUT))
R.to_csv(P + r"\k_pdp.csv", index=False, encoding="utf-8-sig")
