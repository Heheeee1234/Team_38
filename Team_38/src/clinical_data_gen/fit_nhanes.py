"""Independent community-baseline fit from CDC NHANES 2017-2018 BP examination data."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

def age_band(age): return "under_40" if age < 40 else "40_to_64" if age < 65 else "65_plus"
def fit(bp_path: Path, demo_path: Path) -> dict:
    bp=pd.read_sas(bp_path); demo=pd.read_sas(demo_path)[["SEQN","RIDAGEYR","RIAGENDR"]]
    df=bp.merge(demo,on="SEQN"); df=df[df.RIDAGEYR>=18].copy(); df["band"]=df.RIDAGEYR.map(age_band); df["sex"]=df.RIAGENDR.map({1.0:"male",2.0:"female"})
    # Median of repeated seated readings prevents first-reading effects dominating baseline.
    df["heart_rate"]=df[["BPXPLS"]].median(axis=1); df["systolic_bp"]=df[["BPXSY1","BPXSY2","BPXSY3"]].median(axis=1); df["diastolic_bp"]=df[["BPXDI1","BPXDI2","BPXDI3"]].median(axis=1)
    out={}
    for (band_name,sex),g in df.groupby(["band","sex"]):
        out[f"{band_name}_{sex}"]={k:{"median":round(float(g[k].median()),2),"n":int(g[k].notna().sum())} for k in ("heart_rate","systolic_bp","diastolic_bp")}
    return {"fit_version":"nhanes-2017-2018-bpx-v1","source":{"dataset":"CDC NHANES 2017-2018 Blood Pressure (BPX_J)","bp_file":str(bp_path),"demographics_file":str(demo_path),"population":"non-institutionalised US examination participants aged 18+","scope":"community baseline only; not a deterioration trajectory source"},"baseline_by_age_sex":out}
def write_fit(bp:Path,demo:Path,out:Path):
    result=fit(bp,demo); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2),encoding="utf-8"); return result
