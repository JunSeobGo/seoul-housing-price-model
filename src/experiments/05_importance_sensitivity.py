# -*- coding: utf-8 -*-
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import DATA_DIR, RESULTS_DIR
import numpy as np, pandas as pd, xgboost as xgb, json, time
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error
OUT=[]
def say(s=""): OUT.append(str(s)); print(s)
B=DATA_DIR
d=pd.read_csv(B+r"\아파트_매매_전처리.csv")
d=d[(d["계약년"]==2019)&(d["계약분기"]==1)].copy().reset_index(drop=True)
d["단지키"]=d["구"]+"_"+d["동"]+"_"+d["단지명"]; d["단가"]=d["거래금액(만원)"]/d["전용면적(㎡)"]
BF=["전용면적(㎡)","층","건축년도","편의점(개)","약국(개)","병원(개)","공공도서관(개)","경찰서(개)",
    "소방서(개)","대형마트(개)","초등학교 거리(m)","중학교 거리(m)","고등학교 거리(m)",
    "지하철역까지 거리(m)","정류소까지 거리(m)"]

say("="*88); say("[A] 슬라이드 0.48 추적 : 중요도는 하이퍼파라미터에 얼마나 흔들리는가"); say("="*88)
Xs=d[BF]; ys=d["거래금액(만원)"]
Xtr,Xte,ytr,yte=train_test_split(Xs,ys,test_size=0.2,random_state=42)
say("A-1. importance_type 별 (기본 파라미터, 무작위 분할)"); say("-"*88)
m=xgb.XGBRegressor(objective="reg:squarederror",random_state=42).fit(Xtr,ytr)
bst=m.get_booster()
for t in ["gain","weight","cover","total_gain","total_cover"]:
    sc=bst.get_score(importance_type=t); tot=sum(sc.values())
    say("   %-13s 전용면적 %.4f   1순위 %s"%(t,sc.get("전용면적(㎡)",0)/tot,max(sc,key=sc.get)))
say(""); say("A-2. max_depth 별 (gain, n_est=100, lr=0.3)"); say("-"*88)
say("   %-10s %10s %10s  %s"%("max_depth","전용면적","R2","1순위"))
for md in [2,3,4,5,6,8,10]:
    mm=xgb.XGBRegressor(objective="reg:squarederror",random_state=42,max_depth=md).fit(Xtr,ytr)
    s=pd.Series(mm.feature_importances_,index=BF)
    say("   %-10d %10.4f %10.4f  %s"%(md,s["전용면적(㎡)"],r2_score(yte,mm.predict(Xte)),s.idxmax()))
say(""); say("A-3. n_estimators x learning_rate (max_depth=3)"); say("-"*88)
say("   %-8s %-8s %10s %10s"%("n_est","lr","전용면적","R2"))
best=None
for ne in [50,100,300]:
    for lr in [0.05,0.1,0.3]:
        mm=xgb.XGBRegressor(objective="reg:squarederror",random_state=42,max_depth=3,
                            n_estimators=ne,learning_rate=lr).fit(Xtr,ytr)
        s=pd.Series(mm.feature_importances_,index=BF); r2=r2_score(yte,mm.predict(Xte))
        say("   %-8d %-8.2f %10.4f %10.4f"%(ne,lr,s["전용면적(㎡)"],r2))
        c=abs(s["전용면적(㎡)"]-0.48)
        if best is None or c<best[0]: best=(c,ne,lr,float(s["전용면적(㎡)"]),float(r2))
say(""); say("   0.48 최근접: n_est=%d lr=%.2f depth=3 -> 전용면적 %.4f (R2 %.4f)"%(best[1],best[2],best[3],best[4]))

say(""); say("="*88)
say("[B] 튜닝 : log 제곱미터당 단가 / 단지 그룹 5겹 / 변수 15종 + 구 25")
say("="*88)
X=pd.concat([d[BF],pd.get_dummies(d["구"],prefix="구")],axis=1).astype(float)
y=np.log(d["단가"]).values; g=d["단지키"].values
folds=list(GroupKFold(n_splits=5).split(X,y,g))
def oof(params):
    o=np.zeros(len(X))
    for tr,te in folds:
        mm=xgb.XGBRegressor(objective="reg:squarederror",random_state=42,
                            tree_method="hist",n_jobs=4,**params)
        mm.fit(X.iloc[tr],y[tr]); o[te]=mm.predict(X.iloc[te])
    return r2_score(y,o), mean_absolute_error(np.exp(y),np.exp(o)), o
r2b,maeb,_=oof({})
say("기본값   R2 %.4f | MAE %.1f 만원/m2"%(r2b,maeb))
rng=np.random.default_rng(42); t0=time.time(); trials=[]
for i in range(45):
    p={"max_depth":int(rng.integers(3,11)),
       "learning_rate":float(np.exp(rng.uniform(np.log(0.01),np.log(0.3)))),
       "n_estimators":int(rng.integers(150,700)),
       "subsample":float(rng.uniform(0.6,1.0)),
       "colsample_bytree":float(rng.uniform(0.5,1.0)),
       "min_child_weight":int(rng.integers(1,20)),
       "reg_lambda":float(np.exp(rng.uniform(np.log(0.1),np.log(20)))),
       "reg_alpha":float(np.exp(rng.uniform(np.log(1e-3),np.log(5)))),
       "gamma":float(rng.uniform(0,4))}
    r2,mae,_=oof(p); trials.append((mae,r2,p))
trials.sort(key=lambda t:t[0])
maet,r2t,bp=trials[0]
say("튜닝 후  R2 %.4f | MAE %.1f 만원/m2   MAE %+.1f%%   (45회 탐색, %.0f초)"
    %(r2t,maet,-(maet-maeb)/maeb*100,time.time()-t0))
say(""); say("선택된 파라미터"); say("-"*88)
for k,v in bp.items(): say("   %-18s %s"%(k, round(v,4) if isinstance(v,float) else v))
say(""); say("   탐색 45회 중 MAE 분포: 최저 %.1f / 중앙 %.1f / 최고 %.1f"
             %(trials[0][0],trials[len(trials)//2][0],trials[-1][0]))

say(""); say("[C] 튜닝이 변수 순위를 바꾸는가 (동일 분할)"); say("-"*88)
def meanimp(params):
    ims=[]
    for tr,te in folds:
        mm=xgb.XGBRegressor(objective="reg:squarederror",random_state=42,
                            tree_method="hist",n_jobs=4,**params)
        mm.fit(X.iloc[tr],y[tr]); ims.append(pd.Series(mm.feature_importances_,index=X.columns))
    s=pd.concat(ims,axis=1).mean(axis=1)
    loc=s[[c for c in s.index if c.startswith("구_")]].sum()
    s=s[[c for c in s.index if not c.startswith("구_")]]; s["[구 더미 합계]"]=loc
    return s.sort_values(ascending=False)
ib,it=meanimp({}),meanimp(bp)
say("   %-20s %12s %12s"%("변수","기본값","튜닝 후"))
for k in ib.head(8).index: say("   %-20s %12.4f %12.4f"%(k,ib[k],it.get(k,np.nan)))
say(""); say("   상위5 기본값 : "+" > ".join(ib.head(5).index))
say("   상위5 튜닝후 : "+" > ".join(it.head(5).index))
json.dump({"best":bp,"r2_base":r2b,"mae_base":maeb,"r2_tuned":r2t,"mae_tuned":maet},
  open(RESULTS_DIR,"w",encoding="utf-8"),ensure_ascii=False,indent=1)
open(RESULTS_DIR,"w",encoding="utf-8").write("\n".join(OUT))