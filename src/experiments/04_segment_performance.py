# -*- coding: utf-8 -*-
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import DATA_DIR, RESULTS_DIR
import numpy as np, pandas as pd, xgboost as xgb, re
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error
OUT=[]
def say(s=""): OUT.append(str(s)); print(s)
B=DATA_DIR
INF=["편의점(개)","약국(개)","병원(개)","공공도서관(개)","경찰서(개)","소방서(개)","대형마트(개)",
     "초등학교 거리(m)","중학교 거리(m)","고등학교 거리(m)","지하철역까지 거리(m)","정류소까지 거리(m)"]
SEG=[("아파트 매매","아파트_매매_전처리.csv","거래금액(만원)",False),
     ("아파트 전세","아파트_전세_전처리.csv","보증금(만원)",False),
     ("아파트 월세","아파트_월세_전처리.csv","보증금(만원)",True),
     ("연립 매매","연립다세대_매매_전처리.csv","거래금액(만원)",False),
     ("연립 전세","연립다세대_전세_전처리.csv","보증금(만원)",False),
     ("연립 월세","연립다세대_월세_전처리.csv","보증금(만원)",True)]
def mk(): return xgb.XGBRegressor(objective="reg:squarederror",random_state=42)
DIRTY=[]
def clean(d,cols,name):
    for c in cols:
        if d[c].dtype==object:
            bad=pd.to_numeric(d[c],errors="coerce").isna().sum()
            ex=d.loc[pd.to_numeric(d[c],errors="coerce").isna(),c].head(2).tolist()
            DIRTY.append((name,c,int(bad),ex))
            d[c]=pd.to_numeric(d[c],errors="coerce")
    return d
say("="*94)
say("[9] 2019 Q1 · 6개 거래유형 전체 · 슬라이드 조건(무작위 분할) vs 그룹 분할(단지)")
say("="*94)
say("%-11s %6s %9s %8s %11s %10s %-14s"%("세그먼트","n","R2 무작위","R2 그룹","MAE 무작위","MAE 그룹","1순위 변수"))
say("-"*94)
for name,f,tgt,ismon in SEG:
    d=pd.read_csv(B+"\\"+f)
    d=d[(d["계약년"]==2019)&(d["계약분기"]==1)].copy().reset_index(drop=True)
    F=["전용면적(㎡)","층","건축년도"]+INF+(["월세금(만원)"] if ismon else [])
    d=clean(d,F+[tgt],name)
    d=d[d[tgt]>0].dropna(subset=F+[tgt]).reset_index(drop=True)
    X=d[F]; y=d[tgt]; g=d["구"]+"_"+d["동"]+"_"+d["단지명"].astype(str)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42)
    m=mk().fit(Xtr,ytr); pr=m.predict(Xte)
    top=pd.Series(m.feature_importances_,index=F).idxmax()
    oof=np.zeros(len(X))
    for tr,te in GroupKFold(n_splits=5).split(X,y,g):
        oof[te]=mk().fit(X.iloc[tr],y.iloc[tr]).predict(X.iloc[te])
    say("%-11s %6d %9.3f %8.3f %11.0f %10.0f %-14s"
        %(name,len(d),r2_score(yte,pr),r2_score(y,oof),
          mean_absolute_error(yte,pr),mean_absolute_error(y,oof),top))
say("-"*94)
say("MAE 단위 만원. 1순위 변수는 무작위 분할 gain importance 기준.")
say("")
say("[10] 월세 세그먼트에서 월세금 변수를 제거했을 때")
say("-"*94)
for name,f,tgt,ismon in SEG:
    if not ismon: continue
    d=pd.read_csv(B+"\\"+f); d=d[(d["계약년"]==2019)&(d["계약분기"]==1)].copy()
    F0=["전용면적(㎡)","층","건축년도"]+INF
    d=clean(d,F0+["월세금(만원)",tgt],name+"_2")
    d=d[d[tgt]>0].dropna(subset=F0+["월세금(만원)",tgt]).reset_index(drop=True)
    for lab,F in [("월세금 포함",F0+["월세금(만원)"]),("월세금 제외",F0)]:
        Xtr,Xte,ytr,yte=train_test_split(d[F],d[tgt],test_size=0.2,random_state=42)
        m=mk().fit(Xtr,ytr); s=pd.Series(m.feature_importances_,index=F).sort_values(ascending=False)
        say("   %-9s %-9s R2 %.3f | 1순위 %s %.3f | 2순위 %s %.3f"
            %(name,lab,r2_score(yte,m.predict(Xte)),s.index[0],s.iloc[0],s.index[1],s.iloc[1]))
say("")
say("[11] 원본 데이터 품질 이슈 (문자열 혼입 컬럼)")
say("-"*94)
seen=set()
for n,c,b,ex in DIRTY:
    k=(n.split("_")[0],c)
    if k in seen or b==0: continue
    seen.add(k); say("   %-9s %-16s 숫자변환 실패 %d행  예: %s"%(k[0],c,b,ex))
if not seen: say("   없음")
open(RESULTS_DIR,"w",encoding="utf-8").write("\n".join(OUT))