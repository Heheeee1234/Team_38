"""Fit transparent telemetry parameters and real sepsis trajectory templates.

PhysioNet Challenge 2019 has no medication administration or history fields. This
module records medication effects as unidentifiable instead of manufacturing them.
"""
from __future__ import annotations
import csv, json, math
from collections import defaultdict
from pathlib import Path
from typing import Iterable
import numpy as np

CHANNELS = {"heart_rate": "HR", "spo2": "O2Sat", "respiratory_rate": "Resp", "systolic_bp": "SBP", "diastolic_bp": "DBP"}

def band(age: float) -> str: return "under_40" if age < 40 else "40_to_64" if age < 65 else "65_plus"
def f(value: str | None) -> float | None:
    try:
        result=float(value) if value not in (None,"","NaN") else None
        return result if result is not None and math.isfinite(result) else None
    except ValueError: return None
def files(root: Path) -> Iterable[Path]: return root.rglob("*.psv")

def fit(root: Path, max_files: int | None = None) -> dict:
    # Sufficient statistics for age/sex stratified per-patient baselines and AR(1)/OU fits.
    strata: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    # Online sufficient statistics keep a full 40k-file fit memory bounded.
    pairs={name:[0,0.,0.,0.,0.,0.] for name in CHANNELS}; cov_n=0; cov_mean=np.zeros(len(CHANNELS)); cov_m2=np.zeros((len(CHANNELS),len(CHANNELS))); templates=[]; total=0; positive=0
    for path in files(root):
        total += 1
        if max_files and total > max_files: break
        with path.open(newline="",encoding="utf-8") as handle: rows=list(csv.DictReader(handle,delimiter="|"))
        if not rows: continue
        age=f(rows[0].get("Age")); gender=rows[0].get("Gender")
        if age is None or gender not in {"0","1"}: continue
        key=f"{band(age)}_sex_{gender}"; values={name:[] for name in CHANNELS}
        for row in rows:
            for name,column in CHANNELS.items():
                x=f(row.get(column))
                if x is not None: values[name].append(x)
        for name,series in values.items():
            if series: strata[key][name].append(float(np.median(series)))
        # Consecutive observed values only: this avoids treating sparse charting gaps as 1-hour dynamics.
        for name,column in CHANNELS.items():
            previous=None
            for row in rows:
                x=f(row.get(column))
                if x is None: previous=None
                elif previous is not None:
                    stat=pairs[name]; stat[0]+=1; stat[1]+=previous; stat[2]+=x; stat[3]+=previous*previous; stat[4]+=x*x; stat[5]+=previous*x; previous=x
                else: previous=x
        onset=next((i for i,row in enumerate(rows) if f(row.get("SepsisLabel")) == 1.0), None)
        if onset is not None and onset >= 6:
            window=[]
            for row in rows[max(0,onset-6):min(len(rows),onset+6)]:
                window.append([f(row.get(column)) for column in CHANNELS.values()])
            if len(window)==12:
                templates.append(window); positive += 1
        for row in rows:
            row_values=[f(row.get(column)) for column in CHANNELS.values()]
            if all(v is not None for v in row_values):
                cov_n += 1; vector=np.array(row_values,dtype=float); delta=vector-cov_mean; cov_mean += delta/cov_n; cov_m2 += np.outer(delta,vector-cov_mean)
    if not total: raise ValueError(f"No .psv files under {root}")
    global_baselines={}
    by_stratum={}
    for key, channels in strata.items():
        by_stratum[key]={name:{"median":round(float(np.median(vals)),3),"dispersion_mad":round(float(np.median(np.abs(np.array(vals)-np.median(vals)))),3),"n_patients":len(vals)} for name,vals in channels.items() if vals}
    for name in CHANNELS:
        vals=[v for channels in strata.values() for v in channels[name]]
        global_baselines[name]=round(float(np.median(vals)),3)
    ou={}
    for name,stat in pairs.items():
        n,sx,sy,sxx,syy,sxy=stat; varx=sxx-sx*sx/max(n,1); vary=syy-sy*sy/max(n,1); phi=(sxy-sx*sy/max(n,1))/math.sqrt(max(varx*vary,1e-9)) if n>2 else .8
        phi=min(.999,max(.001,phi))
        # AR residual dispersion from variance identity, avoiding materialised pairs.
        residual_std=math.sqrt(max((syy-sy*sy/max(n,1))/max(n-1,1)*(1-phi*phi),0))
        ou[name]={"lag_hours":1,"phi":round(phi,5),"mean_reversion_per_hour":round(-math.log(phi),5),"innovation_std":round(residual_std,4),"n_consecutive_pairs":n}
    covariance=cov_m2/max(cov_n-1,1); scales=np.sqrt(np.maximum(np.diag(covariance),1e-9)); corr=(covariance/np.outer(scales,scales)).round(5).tolist() if cov_n>2 else np.eye(len(CHANNELS)).tolist()
    # Median, baseline-normalised real windows. Keep only fully observed rows for replay templates.
    valid=[np.array(t,dtype=float) for t in templates if np.isfinite(np.array(t,dtype=float)).all()]
    template_payload=[]
    for item in valid[:500]:
        baseline=np.median(item[:3],axis=0); template_payload.append((item-baseline).round(3).tolist())
    return {"fit_version":"physionet-2019-v1","source":{"dataset":"PhysioNet Challenge 2019 training set","path":str(root),"files_scanned":min(total,max_files) if max_files else total,"positive_onset_windows":positive,"template_windows":len(template_payload)},"channels":list(CHANNELS),"baseline_by_age_sex":by_stratum,"global_baseline":global_baselines,"ou":ou,"residual_correlation":corr,"trajectory_templates":{"sepsis_like":{"hours_relative_to_label_onset":list(range(-6,6)),"baseline_normalised_deltas":template_payload}},"medication_effects":{"status":"unidentifiable_from_source","reason":"Challenge 2019 files contain no medication administration/history field; do not apply fitted medication effects."}}

def write_fit(source: Path, target: Path, max_files: int | None = None) -> dict:
    result=fit(source,max_files); target.parent.mkdir(parents=True,exist_ok=True); target.write_text(json.dumps(result,indent=2),encoding="utf-8"); return result
