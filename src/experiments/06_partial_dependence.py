# -*- coding: utf-8 -*-
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import DATA_DIR, RESULTS_DIR
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import GroupKFold
OUT=[]
def say(s=""): OUT.append(str(s)); print(s)
B=DATA_DIR
d=pd.read_csv(B+r"\아파트_매매_전처리.csv")
d=d[(d["계약년"]==2019)&(d["계약분기"]==1)].copy().reset_index(drop=True)
d["단지키"]=d["구"]+"_"+d["동"]+"_"+d["단지명"]; d["단가"]=d["거래금액(만원)"]/d["전용면적(㎡)"]
d["평당가"]=d["단가"]*3.3058
BF=["전용면적(㎡)","층","건축년도","편의점(개)","약국(개)","병원(개)","공공도서관(개)","경찰서(개)",
    "소방서(개)","대형마트(개)","초등학교 거리(m)","중학교 거리(m)","고등학교 거리(m)",
    "지하철역까지 거리(m)","정류소까지 거리(m)"]
X=pd.concat([d[BF],pd.get_dummies(d["구"],prefix="구")],axis=1).astype(float)
y=np.log(d["단가"]).values
BP=dict(max_depth=3,learning_rate=0.0699,n_estimators=448,subsample=0.7068,
        colsample_bytree=0.6658,min_child_weight=9,reg_lambda=1.5779,reg_alpha=0.042,gamma=0.0864)
mdl=xgb.XGBRegressor(objective="reg:squarederror",random_state=42,tree_method="hist",**BP).fit(X,y)

def pdp(col,grid):
    out=[]
    for v in grid:
        Z=X.copy(); Z[col]=v
        out.append(float(np.exp(mdl.predict(Z)).mean()*3.3058))
    return out

say("="*86); say("[D] 부분의존도 : 변수 하나만 바꿨을 때 예측 평당가(만원)"); say("="*86)
say("D-1. 건축년도  (재건축 기대 구간이 있는가)"); say("-"*86)
gr=[1970,1975,1980,1985,1990,1995,2000,2005,2010,2015,2018]
v=pdp("건축년도",gr)
for a,b in zip(gr,v): say("   %d년   %7.0f 만원/평   %s"%(a,b,"#"*int(b/60)))
say("   최저 %d년 / 최고 %d년"%(gr[int(np.argmin(v))],gr[int(np.argmax(v))]))

say(""); say("D-2. 지하철역까지 거리 (역세권 프리미엄이 끊기는 지점)"); say("-"*86)
gr=[100,200,300,400,500,700,1000,1500,2000,3000]
v=pdp("지하철역까지 거리(m)",gr)
for a,b in zip(gr,v): say("   %5dm  %7.0f 만원/평   %s"%(a,b,"#"*int(b/60)))
dv=[v[0]-x for x in v]
say("   100m 대비 하락폭: 500m %.0f만원 / 1000m %.0f만원 / 2000m %.0f만원"%(dv[4],dv[6],dv[8]))

say(""); say("D-3. 전용면적 (규모의 경제)"); say("-"*86)
gr=[30,40,50,60,72,85,100,115,135,165]
v=pdp("전용면적(㎡)",gr)
for a,b in zip(gr,v): say("   %5.0fm2 (%4.1f평)  %7.0f 만원/평"%(a,a/3.3058,b))

say(""); say("="*86); say("[E] 도메인 대조 : 모델 밖 실제 값으로 해석 검증"); say("="*86)
say("E-1. 구별 실제 평당가 상위/하위 (2019 Q1 아파트 매매 중앙값)"); say("-"*86)
gu=d.groupby("구")["평당가"].agg(["median","size"]).sort_values("median",ascending=False)
for i,(k,r) in enumerate(gu.iterrows()):
    if i<6 or i>=len(gu)-4: say("   %2d위 %-6s %7.0f 만원/평  (n=%d)"%(i+1,k,r["median"],r["size"]))
    elif i==6: say("        ...")

say(""); say("E-2. 편의점 밀도가 음의 방향인 이유 확인"); say("-"*86)
d["편의점구간"]=pd.qcut(d["편의점(개)"],4,labels=["하위25%","25-50%","50-75%","상위25%"])
t=d.groupby("편의점구간",observed=True).agg(
    편의점중앙=("편의점(개)","median"),평당가중앙=("평당가","median"),
    건축년도중앙=("건축년도","median"),전용면적중앙=("전용면적(㎡)","median"),n=("평당가","size"))
say(t.round(0).to_string())
top=d[d["편의점구간"]=="상위25%"]["구"].value_counts().head(5)
bot=d[d["편의점구간"]=="하위25%"]["구"].value_counts().head(5)
say("   편의점 상위25% 매물이 몰린 구: "+", ".join("%s %d"%(k,v) for k,v in top.items()))
say("   편의점 하위25% 매물이 몰린 구: "+", ".join("%s %d"%(k,v) for k,v in bot.items()))

say(""); say("E-3. 병원 밀도와 평당가"); say("-"*86)
d["병원구간"]=pd.qcut(d["병원(개)"],4,labels=["하위25%","25-50%","50-75%","상위25%"])
t2=d.groupby("병원구간",observed=True).agg(병원중앙=("병원(개)","median"),
    평당가중앙=("평당가","median"),n=("평당가","size"))
say(t2.round(0).to_string())

say(""); say("E-4. 규모 5분류별 평당가 (규모의 경제 실제 확인)"); say("-"*86)
d["규모"]=pd.cut(d["전용면적(㎡)"],[0,40,60,85,135,1e9],labels=["소형","중소형","중형","중대형","대형"])
t3=d.groupby("규모",observed=True).agg(평당가중앙=("평당가","median"),
    거래금액중앙=("거래금액(만원)","median"),n=("평당가","size"))
say(t3.round(0).to_string())
open(RESULTS_DIR,"w",encoding="utf-8").write("\n".join(OUT))