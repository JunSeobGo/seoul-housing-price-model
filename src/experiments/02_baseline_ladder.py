# -*- coding: utf-8 -*-
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import DATA_DIR, RESULTS_DIR
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
OUT=[]
def say(s=""): OUT.append(str(s)); print(s)

BASE = DATA_DIR
df = pd.read_csv(BASE + r"\아파트_매매_전처리.csv")
d = df[(df["계약년"]==2019)&(df["계약분기"]==1)].copy().reset_index(drop=True)
d["단지키"]=d["구"]+"_"+d["동"]+"_"+d["단지명"]
d["단가"]=d["거래금액(만원)"]/d["전용면적(㎡)"]
d["규모"]=pd.cut(d["전용면적(㎡)"],[0,40,60,85,135,1e9],labels=["소형","중소형","중형","중대형","대형"])
BF=["전용면적(㎡)","층","건축년도","편의점(개)","약국(개)","병원(개)","공공도서관(개)","경찰서(개)",
    "소방서(개)","대형마트(개)","초등학교 거리(m)","중학교 거리(m)","고등학교 거리(m)",
    "지하철역까지 거리(m)","정류소까지 거리(m)"]
y = np.log(d["단가"]); groups=d["단지키"]
gkf=GroupKFold(n_splits=5); folds=list(gkf.split(d,y,groups))

def mk(): return xgb.XGBRegressor(objective="reg:squarederror", random_state=42)

def run(X,tag):
    oof=np.zeros(len(X)); imps=[]
    for tr,te in folds:
        m=mk(); m.fit(X.iloc[tr],y.iloc[tr]); oof[te]=m.predict(X.iloc[te])
        imps.append(pd.Series(m.feature_importances_,index=X.columns))
    imp=pd.concat(imps,axis=1).mean(axis=1)
    return {"tag":tag,"r2":r2_score(y,oof),
            "mae":mean_absolute_error(np.exp(y),np.exp(oof)),"imp":imp,"oof":oof}

X0=d[BF]
Xg=pd.concat([d[BF],pd.get_dummies(d["구"],prefix="구")],axis=1)
Xd=pd.concat([d[BF],pd.get_dummies(d["동"],prefix="동")],axis=1)

R=[run(X0,"M3  기본 15종"),run(Xg,"M4a +구 25"),run(Xd,"M4b +동 259")]

# baselines on same folds
def base_median(keys):
    oof=np.zeros(len(d))
    for tr,te in folds:
        t=d.iloc[tr]; g=t.groupby(keys,observed=True)["단가"].median()
        gl=t["단가"].median()
        for i in te:
            k=tuple(d.iloc[i][c] for c in keys); k=k[0] if len(k)==1 else k
            oof[i]=np.log(g.get(k,gl))
    return oof
def base_mean():
    oof=np.zeros(len(d))
    for tr,te in folds: oof[te]=np.log(d.iloc[tr]["단가"].mean())
    return oof
B=[("B0 전체 평균",base_mean()),("B1 동별 중앙값",base_median(["동"])),
   ("B2 동x규모 중앙값",base_median(["동","규모"]))]

say("="*80)
say("[4] baseline 사다리 + 위치 변수   목표=log 제곱미터당단가, GroupKFold(단지) 5")
say("="*80)
say("%-20s %9s %14s %14s" % ("모델","R2","MAE(만원/m2)","B1대비 개선"))
say("-"*80)
b1mae=None
rows=[]
for n,o in B:
    mae=mean_absolute_error(np.exp(y),np.exp(o)); r2=r2_score(y,o)
    if n.startswith("B1"): b1mae=mae
    rows.append((n,r2,mae))
for r in R: rows.append((r["tag"],r["r2"],r["mae"]))
for n,r2,mae in rows:
    imp = "" if b1mae is None else "%+.1f%%" % (-(mae-b1mae)/b1mae*100)
    say("%-20s %9.4f %14.1f %14s" % (n,r2,mae,imp))

say("")
say("[5] M4b 상위 변수 (동 더미 포함)  * 동 더미는 합산해 위치 한 항목으로 표기")
say("-"*80)
imp=R[2]["imp"].copy()
loc=imp[[c for c in imp.index if c.startswith("동_")]].sum()
imp=imp[[c for c in imp.index if not c.startswith("동_")]]
imp["[동 더미 합계]"]=loc
for k,v in imp.sort_values(ascending=False).head(10).items():
    say("   %-22s %.4f" % (k,v))

say("")
say("[6] 슬라이드 0.48 재현 점검 (거래금액 원값, 무작위 분할, 이상값 처리 유무)")
say("-"*80)
for lab,dd in [("이상값 처리 없음",d),
               ("전용면적/층 1-99% 절단",d.assign(**{
                   "전용면적(㎡)":d["전용면적(㎡)"].clip(*d["전용면적(㎡)"].quantile([.01,.99])),
                   "층":d["층"].clip(*d["층"].quantile([.01,.99]))}))]:
    Xa=dd[BF]; ya=dd["거래금액(만원)"]
    Xtr,Xte,ytr,yte=train_test_split(Xa,ya,test_size=0.2,random_state=42)
    m=mk(); m.fit(Xtr,ytr)
    s=pd.Series(m.feature_importances_,index=BF)
    say("   %-24s 전용면적 %.4f | R2 %.4f | 1순위 %s"
        % (lab,s["전용면적(㎡)"],r2_score(yte,m.predict(Xte)),s.idxmax()))

open(sp:=RESULTS_DIR,"w",encoding="utf-8").write("\n".join(OUT))
np.save(RESULTS_DIR, R[2]["oof"])