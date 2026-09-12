# -*- coding: utf-8 -*-
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import DATA_DIR, RESULTS_DIR
import numpy as np, pandas as pd, xgboost as xgb, shap
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr
OUT=[]
def say(s=""): OUT.append(str(s)); print(s)
BASE=DATA_DIR
df=pd.read_csv(BASE+r"\아파트_매매_전처리.csv")
d=df[(df["계약년"]==2019)&(df["계약분기"]==1)].copy().reset_index(drop=True)
d["단지키"]=d["구"]+"_"+d["동"]+"_"+d["단지명"]; d["단가"]=d["거래금액(만원)"]/d["전용면적(㎡)"]
BF=["전용면적(㎡)","층","건축년도","편의점(개)","약국(개)","병원(개)","공공도서관(개)","경찰서(개)",
    "소방서(개)","대형마트(개)","초등학교 거리(m)","중학교 거리(m)","고등학교 거리(m)",
    "지하철역까지 거리(m)","정류소까지 거리(m)"]
X=pd.concat([d[BF],pd.get_dummies(d["구"],prefix="구")],axis=1)
y=np.log(d["단가"])
folds=list(GroupKFold(n_splits=5).split(X,y,d["단지키"]))
sv=np.zeros((len(X),X.shape[1]))
for tr,te in folds:
    m=xgb.XGBRegressor(objective="reg:squarederror",random_state=42).fit(X.iloc[tr],y.iloc[tr])
    sv[te]=shap.TreeExplainer(m).shap_values(X.iloc[te])
S=pd.DataFrame(sv,columns=X.columns)

say("="*80); say("[7] SHAP  M4a(기본15종+구), log 제곱미터당단가, out-of-fold")
say("="*80)
say("%-20s %12s %10s %s" % ("변수","평균|SHAP|","방향","해석"))
say("-"*80)
rows=[]
for c in BF:
    ms=np.abs(S[c]).mean(); rho=spearmanr(X[c],S[c]).statistic
    rows.append((c,ms,rho))
gu=np.abs(S[[c for c in X.columns if c.startswith("구_")]].sum(axis=1)).mean()
for c,ms,rho in sorted(rows,key=lambda r:-r[1]):
    dirn = "값↑→가격↑" if rho>0.1 else ("값↑→가격↓" if rho<-0.1 else "혼재")
    note=""
    if "거리" in c and rho>0.1: note="  <= 멀수록 비쌈, 통념과 반대"
    if "거리" in c and rho<-0.1: note="  <= 가까울수록 비쌈"
    say("%-20s %12.4f %10s %s (rho %+.2f)%s" % (c,ms,dirn,"",rho,note))
say("%-20s %12.4f %10s" % ("[구 더미 합계]",gu,"-"))

say(""); say("[8] 슬라이드 '편의점 vs 병원' 항목 점검")
say("-"*80)
for c in ["편의점(개)","병원(개)"]:
    say("   %-10s 평균|SHAP| %.4f | rho %+.2f | 방향 %s"
        % (c,np.abs(S[c]).mean(),spearmanr(X[c],S[c]).statistic,
           "가격 상승" if spearmanr(X[c],S[c]).statistic>0 else "가격 하락"))
open(RESULTS_DIR,"w",encoding="utf-8").write("\n".join(OUT))