"""Causal synthetic trajectory generator; labels never enter observed events."""
from __future__ import annotations

import hashlib
import json
import math
import random
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
import numpy as np

from . import SCHEMA_VERSION
from .schema import EventEnvelope


@dataclass
class Profile:
    patient_id: str; age: int; sex: str; history: list[str]; medications: list[str]
    labs: dict[str, float]; baseline: dict[str, float]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class ClinicalDataGenerator:
    def __init__(self, seed: int, config: dict[str, Any], fitted_params: dict[str, Any] | None = None) -> None:
        self.seed, self.rng, self.config = seed, random.Random(seed), config
        self.run_id = f"synthetic-{seed}-{uuid.uuid5(uuid.NAMESPACE_DNS, str(seed)).hex[:10]}"; self.fitted_params=fitted_params or {}; self.np_rng=np.random.default_rng(seed)
        self.channels=("heart_rate","spo2","respiratory_rate","systolic_bp","diastolic_bp")
        ou=self.fitted_params.get("ou",{}); self.phi=np.array([ou.get(c,{}).get("phi",.7) for c in self.channels]); self.innovation=np.array([ou.get(c,{}).get("innovation_std",self.config["noise"][c]) for c in self.channels])
        corr=np.array(self.fitted_params.get("residual_correlation",np.eye(5)),dtype=float)
        try: self.correlation_cholesky=np.linalg.cholesky(corr+np.eye(5)*1e-6)
        except np.linalg.LinAlgError: self.correlation_cholesky=np.eye(5)

    def profile(self, ordinal: int) -> Profile:
        age, sex, history, medications = self.rng.randint(25, 92), self.rng.choice(["female","male"]), [], []
        if self.rng.random() < .28: history.append("COPD")
        if self.rng.random() < .33: history.append("hypertension")
        if self.rng.random() < .18: history.append("chronic kidney disease")
        if self.rng.random() < .22: medications.append("beta blocker")
        if "hypertension" in history: medications.append("antihypertensive")
        key=f"{'under_40' if age<40 else '40_to_64' if age<65 else '65_plus'}_sex_{'0' if sex=='female' else '1'}"; fit=self.fitted_params.get("baseline_by_age_sex",{}).get(key,{})
        default={"heart_rate":72,"spo2":96.5,"respiratory_rate":16,"systolic_bp":122,"diastolic_bp":76}
        base={name:fit.get(name,{}).get("median",value) for name,value in default.items()}
        if "COPD" in history: base["spo2"]=min(base["spo2"],91.5); base["respiratory_rate"]+=2
        # Medication effects are intentionally not applied: PhysioNet 2019 cannot identify them.
        return Profile(f"SYN-{ordinal:04d}", age, sex, history or ["none recorded"], medications or ["none recorded"], {"creatinine_mg_dl": round(.7+self.rng.random()*.8+(.5 if "chronic kidney disease" in history else 0),2), "wbc_k_ul": round(4.5+self.rng.random()*6,1)}, base)

    def _delta(self, scenario: str, p: float, template: list[list[float]] | None = None) -> dict[str, float]:
        if template and scenario in {"sepsis_like", "beta_blocked_sepsis"}:
            row=template[min(len(template)-1, round(p*(len(template)-1)))]; values={k:row[i] for i,k in enumerate(("heart_rate","spo2","respiratory_rate","systolic_bp","diastolic_bp"))}
            if scenario == "beta_blocked_sepsis": values["heart_rate"] *= .3
            return values
        values = {"respiratory_decline": (24,-8,12,-5,-3), "sepsis_like": (30,-4,10,-18,-10), "beta_blocked_sepsis": (8,-4,10,-18,-10), "haemodynamic_shock": (35,-2,7,-32,-18), "opioid_respiratory_depression": (-12,-7,-8,-12,-8), "masked_hypoxemia": (18,-1,10,-4,-2), "post_physiotherapy": (16,0,6,2,1), "anxiety": (18,0,7,4,3), "ambulation": (22,0,8,5,3)}
        return dict(zip(("heart_rate","spo2","respiratory_rate","systolic_bp","diastolic_bp"), (x*p for x in values.get(scenario, (0,0,0,0,0)))))

    def create(self, patients: int, minutes: int, start: datetime | None = None) -> dict[str, list[dict[str, Any]]]:
        start, cadence = start or datetime(2026,9,18,8,0,tzinfo=UTC), int(self.config["cadence_seconds"])
        steps = max(2, minutes*60//cadence); observed=[]; contexts=[]; labels=[]; episodes=[]
        weights = self.config["scenario_weights"]
        for ordinal in range(1, patients+1):
            profile = self.profile(ordinal); scenario = self.rng.choices(list(weights), weights=list(weights.values()))[0]
            templates=self.fitted_params.get("trajectory_templates",{}).get("sepsis_like",{}).get("baseline_normalised_deltas",[])
            template=self.rng.choice(templates) if scenario in {"sepsis_like","beta_blocked_sepsis"} and templates else None
            onset = self.rng.randint(max(3,steps//4), max(4,steps//2))
            contexts.append(self._context(profile,start))
            residual=np.zeros(5)
            for index in range(steps):
                p = 0 if scenario in {"stable","artifact_only"} or index < onset else clamp((index-onset)/max(1,steps-onset-1),0,1)
                if scenario in {"post_physiotherapy","anxiety","ambulation"}: p *= math.exp(-max(0,index-onset)/5)
                latent={k:round(v+self._delta(scenario,p,template)[k],2) for k,v in profile.baseline.items()}
                vitals, artifact, residual=self._observe(latent,scenario,index,onset,residual)
                at=start+timedelta(seconds=index*cadence); eid=str(uuid.uuid5(uuid.NAMESPACE_URL,f"{self.run_id}:{profile.patient_id}:{index}"))
                observed.append(EventEnvelope(SCHEMA_VERSION,eid,at.isoformat(),profile.patient_id,self.run_id,"synthetic-bedside-monitor",{"vitals":vitals,"sequence":index,"quality":"suspect" if artifact else "normal"}).to_dict())
                labels.append({"event_id":eid,"patient_id":profile.patient_id,"event_time":at.isoformat(),"scenario":scenario,"trend_phase":"deteriorating" if p>=.45 else "baseline","artifact_injected":artifact,"latent_vitals":latent})
            if scenario not in {"stable","artifact_only","post_physiotherapy","anxiety","ambulation"}:
                ei=onset+math.ceil((steps-onset)*.45)
                evidence={"respiratory_decline":["falling SpO2","rising respiratory rate","compensatory tachycardia"],"sepsis_like":["rising heart rate","rising respiratory rate","falling blood pressure"],"beta_blocked_sepsis":["rising respiratory rate","falling blood pressure","blunted heart-rate response"],"haemodynamic_shock":["falling blood pressure","rising heart rate","sustained multi-reading trend"],"opioid_respiratory_depression":["falling respiratory rate","falling SpO2","falling heart rate"],"masked_hypoxemia":["rising respiratory rate","tachycardia","increased supplemental oxygen despite near-normal SpO2"]}[scenario]
                episodes.append({"episode_id":f"EP-{self.run_id[-10:]}-{ordinal:04d}","patient_id":profile.patient_id,"scenario":scenario,"onset_time":(start+timedelta(seconds=onset*cadence)).isoformat(),"escalation_window_start":(start+timedelta(seconds=ei*cadence)).isoformat(),"expected_evidence":evidence,"label_source":"generator causal scenario"})
        return {"patient_context":contexts,"vitals_observed":observed,"evaluation_labels":labels,"ground_truth_episodes":episodes}

    def _context(self,p: Profile,at: datetime) -> dict[str,Any]:
        return EventEnvelope(SCHEMA_VERSION,str(uuid.uuid5(uuid.NAMESPACE_URL,f"{self.run_id}:{p.patient_id}:context")),at.isoformat(),p.patient_id,self.run_id,"synthetic-ehr-profile",{"age":p.age,"sex":p.sex,"relevant_history":p.history,"current_medications":p.medications,"recent_labs":p.labs}).to_dict()

    def _observe(self, latent: dict[str,float], scenario: str, index: int, onset: int, previous: np.ndarray) -> tuple[dict[str,float],bool,np.ndarray]:
        # Fitted multivariate AR(1): correlation is in the innovations, not post-hoc values.
        residual=self.phi*previous + self.correlation_cholesky @ self.np_rng.normal(size=5) * self.innovation
        result={key:latent[key]+residual[i] for i,key in enumerate(self.channels)}; artifact=scenario=="artifact_only" and index==onset
        if artifact: result["spo2"]-=10; result["heart_rate"]+=14
        bounds={"heart_rate":(30,220),"spo2":(70,100),"respiratory_rate":(5,60),"systolic_bp":(55,230),"diastolic_bp":(30,140)}
        for k,(lo,hi) in bounds.items(): result[k]=round(clamp(result[k],lo,hi),1)
        # Correlated residuals can otherwise invert SBP/DBP at the bounds.
        result["systolic_bp"] = max(result["systolic_bp"], round(result["diastolic_bp"] + 5, 1))
        result["map"]=round((result["systolic_bp"]+2*result["diastolic_bp"])/3,1); result["fio2"] = .21 if scenario != "masked_hypoxemia" or index < onset else round(.21+.35*clamp((index-onset)/30,0,1),2)
        return result,artifact,residual


def write_jsonl(path: Path, records: list[dict[str,Any]]) -> None:
    path.write_text("".join(json.dumps(record,sort_keys=True)+"\n" for record in records),encoding="utf-8")


def sha256(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
