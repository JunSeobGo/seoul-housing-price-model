# -*- coding: utf-8 -*-
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import DATA_DIR, RESULTS_DIR
import sys, io, json
import pandas as pd, numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error

OUT = []
def say(s=""):
    OUT.append(str(s)); print(s)

BASE = DATA_DIR
df = pd.read_csv(BASE + r"\아파트_매매_전처리.csv")
d = df[(df["계약년"]==2019) & (df["계약분기"]==1)].copy().reset_index(drop=True)

d["단지키"] = d["구"]+"_"+d["동"]+"_"+d["단지명"]
d["단가"] = d["거래금액(만원)"] / d["전용면적(㎡)"]

def size_cls(a):
    if a < 40: return "소형"
    if a < 60: return "중소형"
    if a < 85: return "중형"
    if a < 135: return "중대형"
    return "대형"
d["규모"] = d["전용면적(㎡)"].apply(size_cls)

BASE_FEATS = ["전용면적(㎡)","층","건축년도",
    "편의점(개)","약국(개)","병원(개)","공공도서관(개)","경찰서(개)","소방서(개)","대형마트(개)",
    "초등학교 거리(m)","중학교 거리(m)","고등학교 거리(m)","지하철역까지 거리(m)","정류소까지 거리(m)"]

say("="*78)
say("2019 Q1 아파트 매매 재실험  |  n=%d  단지=%d  동=%d  구=%d"
    % (len(d), d["단지키"].nunique(), d["동"].nunique(), d["구"].nunique()))
say("기준금리 고유값 %s (단일 시점이라 상수) / 매매지수 고유값 %d (구 단위 집계값)"
    % (list(d["기준금리"].unique()), d["매매지수"].nunique()))
say("두 변수 모두 독립변수에서 제외. 슬라이드 변수 목록과 동일한 15종만 사용.")
say("="*78)

def mk_model():
    return xgb.XGBRegressor(objective="reg:squarederror", random_state=42)

def imp_series(model, cols):
    return pd.Series(model.feature_importances_, index=cols)

def run_random(X, y, tag, back=None):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    m = mk_model(); m.fit(Xtr, ytr)
    p = m.predict(Xte)
    r2 = r2_score(yte, p)
    if back == "exp":
        mae = mean_absolute_error(np.exp(yte), np.exp(p))
    else:
        mae = mean_absolute_error(yte, p)
    return {"tag":tag, "split":"random 80/20", "r2":r2, "mae":mae,
            "imp":imp_series(m, list(X.columns))}

def run_group(X, y, groups, tag, back=None):
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(X)); imps=[]
    for tr, te in gkf.split(X, y, groups):
        m = mk_model(); m.fit(X.iloc[tr], y.iloc[tr])
        oof[te] = m.predict(X.iloc[te])
        imps.append(imp_series(m, list(X.columns)))
    r2 = r2_score(y, oof)
    if back == "exp":
        mae = mean_absolute_error(np.exp(y), np.exp(oof))
    else:
        mae = mean_absolute_error(y, oof)
    return {"tag":tag, "split":"GroupKFold(단지) 5", "r2":r2, "mae":mae,
            "imp":pd.concat(imps, axis=1).mean(axis=1), "oof":oof}

X_base = d[BASE_FEATS]
groups = d["단지키"]

res = []
# M1 : 슬라이드 재현 시도 (원 코드 그대로 = 거래금액 원값, 무작위 분할)
res.append(run_random(X_base, d["거래금액(만원)"], "M1 거래금액 원값"))
# M1L : 슬라이드 전처리 기술대로 로그변환
res.append(run_random(X_base, np.log(d["거래금액(만원)"]), "M1L log 거래금액", back="exp"))
# M2 : 분할만 그룹으로
res.append(run_group(X_base, np.log(d["거래금액(만원)"]), groups, "M2 log 거래금액", back="exp"))
# M3 : 종속변수를 단가로
res.append(run_group(X_base, np.log(d["단가"]), groups, "M3 log 제곱미터당단가", back="exp"))

say("")
say("[1] 슬라이드 재현과 검증 설계 교체")
say("-"*78)
say("%-24s %-22s %9s %14s %10s" % ("모델","분할","R2","MAE","전용면적중요도"))
for r in res:
    unit = "만원/m2" if "단가" in r["tag"] else "만원"
    say("%-24s %-22s %9.4f %10.1f %s %9.4f"
        % (r["tag"], r["split"], r["r2"], r["mae"], unit, r["imp"]["전용면적(㎡)"]))

json.dump({r["tag"]: {"r2":float(r["r2"]), "mae":float(r["mae"]),
                      "imp":{k:float(v) for k,v in r["imp"].items()}} for r in res},
          open(r"%s\res_part1.json" % RESULTS_DIR,
               "w", encoding="utf-8"), ensure_ascii=False, indent=1)

say("")
say("[2] M3 상위 변수 (log 제곱미터당단가, 그룹분할)")
say("-"*78)
for k,v in res[3]["imp"].sort_values(ascending=False).head(10).items():
    say("   %-22s %.4f" % (k, v))

say("")
say("[3] M1 상위 변수 (슬라이드 조건)")
say("-"*78)
for k,v in res[0]["imp"].sort_values(ascending=False).head(10).items():
    say("   %-22s %.4f" % (k, v))

txt = "\n".join(OUT)
open(RESULTS_DIR,"w",encoding="utf-8").write(txt)