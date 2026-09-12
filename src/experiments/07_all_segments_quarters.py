# -*- coding: utf-8 -*-
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import DATA_DIR, RESULTS_DIR
import numpy as np, pandas as pd, xgboost as xgb, gc
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error
OUT=[]
def say(s=""): OUT.append(str(s)); print(s)
B=DATA_DIR
INF=["편의점(개)","약국(개)","병원(개)","공공도서관(개)","경찰서(개)","소방서(개)","대형마트(개)",
     "초등학교 거리(m)","중학교 거리(m)","고등학교 거리(m)"]
KEEP=["시군구","단지명","전용면적(㎡)","층","건축년도","년도","분기","거리(m)","최단거리(m)",
      "거래금액(만원)","보증금(만원)","월세금(만원)","전월세구분"]+INF
TP=[(2019,1),(2020,2),(2021,4),(2022,4)]
def mk(): return xgb.XGBRegressor(objective="reg:squarederror",random_state=42,tree_method="hist",n_jobs=4)
def num(s):
    if s.dtype==object:
        s=s.astype(str).str.replace(",","",regex=False).str.strip()
    return pd.to_numeric(s,errors="coerce")
def load(f):
    d=pd.read_csv(B+"\\"+f,encoding="CP949",low_memory=False,usecols=lambda c:c in KEEP)
    d=d.rename(columns={"거리(m)":"지하철역까지 거리(m)","최단거리(m)":"정류소까지 거리(m)"})
    sp=d["시군구"].str.split(); d["구"]=sp.str[1]; d["동"]=sp.str[2]
    d["단지키"]=d["구"]+"_"+d["동"]+"_"+d["단지명"].astype(str)
    NUMCOLS=(["전용면적(㎡)","층","건축년도","년도","분기","거래금액(만원)","보증금(만원)",
              "월세금(만원)","지하철역까지 거리(m)","정류소까지 거리(m)"]+INF)
    for c in NUMCOLS:
        if c in d.columns: d[c]=num(d[c])
    return d
SEG=[("아파트 전세","아파트_전월세.csv","전세","보증금(만원)"),
     ("아파트 월세","아파트_전월세.csv","월세","보증금(만원)"),
     ("연립 매매","연립다세대_매매.csv",None,"거래금액(만원)"),
     ("연립 전세","연립다세대_전월세.csv","전세","보증금(만원)"),
     ("연립 월세","연립다세대_전월세.csv","월세","보증금(만원)")]
say("="*100); say("[G] 전처리 재구축 : 4개 분석 시점 x 5개 거래유형"); say("="*100)
say("아파트 매매는 원본 병합 파일에서 2020~2023년 전용면적이 100% 결측이라 제외")
say("금액 컬럼의 천단위 콤마를 제거한 뒤 수치 변환")
say("")
cache={}; rows=[]
for name,f,gubun,tgt in SEG:
    if f not in cache: cache[f]=load(f)
    d=cache[f]
    if gubun: d=d[d["전월세구분"]==gubun]
    F=["전용면적(㎡)","층","건축년도"]+INF+["지하철역까지 거리(m)","정류소까지 거리(m)"]
    if gubun=="월세": F=F+["월세금(만원)"]
    s0=d.dropna(subset=F+[tgt])
    s0=s0[(s0[tgt]>0)&(s0["전용면적(㎡)"]>0)]
    say("   %-10s 정제 후 %d행 (원본 %d행)"%(name,len(s0),len(d)))
    for y,q in TP:
        s=s0[(s0["년도"]==y)&(s0["분기"]==q)].reset_index(drop=True)
        if len(s)<300: rows.append((name,y,q,len(s),None,None,None,None,None,None)); continue
        Xtr,Xte,ytr,yte=train_test_split(s[F],s[tgt],test_size=0.2,random_state=42)
        m=mk().fit(Xtr,ytr); si=pd.Series(m.feature_importances_,index=F)
        yl=np.log(s[tgt]/s["전용면적(㎡)"]).values; g=s["단지키"].values
        X=s[F]; o=np.zeros(len(X)); ims=[]
        for tr,te in GroupKFold(n_splits=5).split(X,yl,g):
            mm=mk().fit(X.iloc[tr],yl[tr]); o[te]=mm.predict(X.iloc[te])
            ims.append(pd.Series(mm.feature_importances_,index=F))
        ic=pd.concat(ims,axis=1).mean(axis=1)
        rows.append((name,y,q,len(s),float(si["전용면적(㎡)"]),r2_score(yte,m.predict(Xte)),si.idxmax(),
                     r2_score(yl,o),float(ic["전용면적(㎡)"]),ic.sort_values(ascending=False).index[0]))
    gc.collect()
say(""); say("G-1. 슬라이드 조건 (원값 종속변수 / 무작위 분할) vs 교정 설계 (log 단가 / 단지 그룹)")
say("-"*100)
say("%-10s %-9s %8s | %8s %7s %-14s | %8s %7s %-14s"
    %("세그먼트","시점","표본","전용면적","R2","1순위","전용면적","R2","1순위"))
for r in rows:
    if r[4] is None: say("%-10s %dQ%-7d %8d | 표본 부족"%(r[0],r[1],r[2],r[3])); continue
    say("%-10s %dQ%-7d %8d | %8.3f %7.3f %-14s | %8.3f %7.3f %-14s"
        %(r[0],r[1],r[2],r[3],r[4],r[5],r[6][:13],r[8],r[7],r[9][:13]))
df=pd.DataFrame(rows,columns=["세그먼트","년","분기","n","면적_슬","R2_슬","1순위_슬","R2_교","면적_교","1순위_교"]).dropna(subset=["면적_슬"])
say(""); say("G-2. 시점 간 1순위 변수 안정성"); say("-"*100)
for seg,g_ in df.groupby("세그먼트",sort=False):
    say("   %-10s 슬라이드 %-46s | 교정 %s"%(seg," > ".join(g_["1순위_슬"]), " > ".join(g_["1순위_교"])))
a=sum(g_["1순위_슬"].nunique()==1 for _,g_ in df.groupby("세그먼트"))
b=sum(g_["1순위_교"].nunique()==1 for _,g_ in df.groupby("세그먼트"))
say(""); say("   4시점 내내 1순위 동일 : 슬라이드 조건 %d/%d, 교정 설계 %d/%d"%(a,df['세그먼트'].nunique(),b,df['세그먼트'].nunique()))
say(""); say("G-3. 전용면적 중요도 시점 간 변동폭"); say("-"*100)
say("   %-10s %26s %26s"%("세그먼트","슬라이드 조건","교정 설계"))
for seg,g_ in df.groupby("세그먼트",sort=False):
    say("   %-10s %11.3f ~ %-11.3f %11.3f ~ %-11.3f"
        %(seg,g_["면적_슬"].min(),g_["면적_슬"].max(),g_["면적_교"].min(),g_["면적_교"].max()))
df.to_csv(RESULTS_DIR,index=False,encoding="utf-8-sig")
open(RESULTS_DIR,"w",encoding="utf-8").write("\n".join(OUT))