"""Reference-vs-synthetic fidelity measurements and a transparent transfer baseline."""
from __future__ import annotations
import csv, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
from .fit_physionet import CHANNELS, f

def ks(a,b):
    a,b=np.sort(a),np.sort(b); grid=np.sort(np.concatenate((a,b)))
    return float(np.max(np.abs(np.searchsorted(a,grid,side="right")/len(a)-np.searchsorted(b,grid,side="right")/len(b))))
def auc(y,s):
    y,s=np.asarray(y),np.asarray(s); pos=y==1; npos=pos.sum(); nneg=len(y)-npos
    if not npos or not nneg:return None
    ranks=np.argsort(np.argsort(s))+1
    return float((ranks[pos].sum()-npos*(npos+1)/2)/(npos*nneg))
def gbm_fit(x,y,rounds=35,learning_rate=.12):
    """Deterministic logistic-loss gradient-boosted decision stumps."""
    base=float(np.log((y.mean()+1e-4)/(1-y.mean()+1e-4))); score=np.full(len(y),base); trees=[]
    for _ in range(rounds):
        residual=y-1/(1+np.exp(-np.clip(score,-30,30))); best=None
        for feature in range(x.shape[1]):
            for threshold in np.unique(np.quantile(x[:,feature],np.linspace(.05,.95,19))):
                left=x[:,feature]<=threshold; right=~left
                if left.sum()<10 or right.sum()<10: continue
                lv,rv=residual[left].mean(),residual[right].mean(); loss=((residual[left]-lv)**2).sum()+((residual[right]-rv)**2).sum()
                if best is None or loss<best[0]: best=(loss,feature,float(threshold),float(lv),float(rv))
        _,feature,threshold,lv,rv=best; update=np.where(x[:,feature]<=threshold,lv,rv); score+=learning_rate*update; trees.append((feature,threshold,lv,rv))
    return base,learning_rate,trees
def predict(x,model):
    base,rate,trees=model; score=np.full(len(x),base)
    for feature,threshold,lv,rv in trees: score+=rate*np.where(x[:,feature]<=threshold,lv,rv)
    return 1/(1+np.exp(-np.clip(score,-30,30)))
def _tree(residual,x,idx,depth,max_depth):
    if depth >= max_depth or len(idx) < 20: return float(residual[idx].mean())
    best=None
    for feature in range(x.shape[1]):
        for threshold in np.unique(np.quantile(x[idx,feature],np.linspace(.1,.9,9))):
            left=idx[x[idx,feature]<=threshold]; right=idx[x[idx,feature]>threshold]
            if len(left)<10 or len(right)<10: continue
            loss=((residual[left]-residual[left].mean())**2).sum()+((residual[right]-residual[right].mean())**2).sum()
            if best is None or loss<best[0]: best=(loss,feature,float(threshold),left,right)
    if best is None:return float(residual[idx].mean())
    _,feature,threshold,left,right=best
    return (feature,threshold,_tree(residual,x,left,depth+1,max_depth),_tree(residual,x,right,depth+1,max_depth))
def _tree_predict(x,tree):
    if not isinstance(tree,tuple): return np.full(len(x),tree)
    feature,threshold,left,right=tree; mask=x[:,feature]<=threshold; out=np.empty(len(x)); out[mask]=_tree_predict(x[mask],left); out[~mask]=_tree_predict(x[~mask],right); return out
def depth3_gbm_fit(x,y,rounds=30,learning_rate=.12):
    """Slightly stronger deterministic logistic-loss gradient-boosted depth-3 trees."""
    base=float(np.log((y.mean()+1e-4)/(1-y.mean()+1e-4))); score=np.full(len(y),base); trees=[]
    for _ in range(rounds):
        residual=y-1/(1+np.exp(-np.clip(score,-30,30))); tree=_tree(residual,x,np.arange(len(x)),0,3); score+=learning_rate*_tree_predict(x,tree); trees.append(tree)
    return base,learning_rate,trees
def depth3_predict(x,model):
    base,rate,trees=model; score=np.full(len(x),base)
    for tree in trees: score+=rate*_tree_predict(x,tree)
    return 1/(1+np.exp(-np.clip(score,-30,30)))
def real_rows(root,max_files=2500):
    data=[]
    for i,path in enumerate(root.rglob("*.psv")):
        if i>=max_files:break
        with path.open(newline="",encoding="utf-8") as h:
            for row in csv.DictReader(h,delimiter="|"):
                values=[f(row.get(c)) for c in CHANNELS.values()]
                if all(v is not None for v in values): data.append((path.stem,values,int(f(row.get("SepsisLabel")) or 0)))
    return data
def synthetic_rows(folder):
    labels={x["event_id"]:x for x in (json.loads(l) for l in (folder/"evaluation_labels.jsonl").read_text().splitlines())}
    out=[]
    for line in (folder/"vitals_observed.jsonl").read_text().splitlines():
        e=json.loads(line); v=e["payload"]["vitals"]; l=labels[e["event_id"]]
        out.append((e["patient_id"],[v[c] for c in CHANNELS],int(l["trend_phase"]=="deteriorating")))
    return out
def patient_split(rows,train=True): return [r for r in rows if (hash(r[0])%5!=0)==train]
def report(reference:Path, generated:Path, params:Path, max_files=2500):
    real=real_rows(reference,max_files); syn=synthetic_rows(generated)
    rv=np.array([r[1] for r in real]); sv=np.array([r[1] for r in syn]); metrics={}
    for i,name in enumerate(CHANNELS):
        metrics[name]={"ks_distance":round(ks(rv[:,i],sv[:,i]),4)}
    rc=np.corrcoef(rv,rowvar=False); sc=np.corrcoef(sv,rowvar=False)
    # Lag-1 ACF measured within patient streams.
    def acf(rows):
        grouped=defaultdict(list)
        for p,x,_ in rows: grouped[p].append(x)
        return [float(np.corrcoef(np.array([v[i] for g in grouped.values() for v in zip(g[:-1],g[1:])]),rowvar=False)[0,1]) if False else 0 for i in range(5)]
    # calculate explicit consecutive series correlation per channel
    def lag(rows,i):
        g=defaultdict(list)
        for p,x,_ in rows:g[p].append(x[i])
        a=np.array([x for s in g.values() for x in s[:-1]]); b=np.array([x for s in g.values() for x in s[1:]])
        return float(np.corrcoef(a,b)[0,1])
    acf_error={name:{"real_lag1":round(lag(real,i),4),"synthetic_lag1":round(lag(syn,i),4),"absolute_error":round(abs(lag(real,i)-lag(syn,i)),4)} for i,name in enumerate(CHANNELS)}
    real_train,real_test=patient_split(real,True),patient_split(real,False); syn_train=patient_split(syn,True)
    def xy(rows):return np.array([r[1] for r in rows]),np.array([r[2] for r in rows])
    xrtr,yrtr=xy(real_train); xt,yt=xy(real_test); xs,ys=xy(syn_train)
    trtr=auc(yt,predict(xt,gbm_fit(xrtr,yrtr)))
    tstr=auc(yt,predict(xt,gbm_fit(xs,ys)))
    trtr_d3=auc(yt,depth3_predict(xt,depth3_gbm_fit(xrtr,yrtr)))
    tstr_d3=auc(yt,depth3_predict(xt,depth3_gbm_fit(xs,ys)))
    labels={x["event_id"]:x for x in (json.loads(l) for l in (generated/"evaluation_labels.jsonl").read_text().splitlines())}
    residuals=[]
    for line in (generated/"vitals_observed.jsonl").read_text().splitlines():
        e=json.loads(line); observed=e["payload"]["vitals"]; latent=labels[e["event_id"]]["latent_vitals"]; residuals.append([observed[c]-latent[c] for c in CHANNELS])
    target=np.array(json.loads(params.read_text())["residual_correlation"],dtype=float); realised=np.corrcoef(np.array(residuals),rowvar=False)
    return {"reference":{"dataset":"PhysioNet Challenge 2019","files_used":max_files,"complete_rows":len(real)},"synthetic_complete_rows":len(syn),"ks_distance":metrics,"acf_lag1":acf_error,"real_vs_synthetic_correlation_frobenius_error":round(float(np.linalg.norm(rc-sc,"fro")),4),"residual_correlation_realisation":{"target_matrix":target.round(4).tolist(),"realised_matrix":realised.round(4).tolist(),"frobenius_error":round(float(np.linalg.norm(target-realised,"fro")),4)},"tstr_vs_trtr":{"task":"synthetic deterioration label transfer to PhysioNet SepsisLabel; labels are not clinically interchangeable","patient_split":"patient-level 80/20","stump_gbm":{"model":"deterministic logistic-loss gradient-boosted decision stumps over five vitals","trtr_auroc":None if trtr is None else round(trtr,4),"tstr_auroc":None if tstr is None else round(tstr,4),"transfer_gap":None if trtr is None or tstr is None else round(trtr-tstr,4)},"depth3_gbm":{"model":"deterministic logistic-loss gradient-boosted depth-3 trees over five vitals","trtr_auroc":None if trtr_d3 is None else round(trtr_d3,4),"tstr_auroc":None if tstr_d3 is None else round(tstr_d3,4),"transfer_gap":None if trtr_d3 is None or tstr_d3 is None else round(trtr_d3-tstr_d3,4)}}}
