from __future__ import annotations
from dataclasses import asdict
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Mapping

from .schema import *

CORE_TO_GENERATED={"HR":"heart_rate","O2Sat":"spo2","Resp":"respiratory_rate","SBP":"systolic_bp","DBP":"diastolic_bp","MAP":"map"}
GENERATED_TO_CORE={v:k for k,v in CORE_TO_GENERATED.items()}

def _value(x):
    if isinstance(x,Mapping): return x.get("value")
    return x

def patient_context(payload: Mapping[str,Any])->PatientContext:
    return PatientContext(age=payload.get("age"),sex=payload.get("sex"),relevant_history=list(payload.get("relevant_history") or []),current_medications=list(payload.get("current_medications") or []),recent_labs=dict(payload.get("recent_labs") or {}))

def trend_evidence(result: Any)->TrendEvidence:
    if result is None: return TrendEvidence()
    if hasattr(result,"to_dict"): result=result.to_dict()
    if isinstance(result,Mapping):
        score=result.get("score", 1.0 if result.get("candidate_deterioration_trend") else 0.0)
        direction=result.get("direction", "deteriorating" if result.get("candidate_deterioration_trend") else "stable")
        signals=result.get("contributing_signals") or result.get("concurrent_worsening_signals") or []
        details=result.get("details") or {x.get("vital"):x.get("change") for x in result.get("trends",[]) if isinstance(x,Mapping)}
        return TrendEvidence(float(score or 0),str(direction),bool(result.get("candidate_deterioration_trend") or result.get("persistent_deterioration",False)),list(signals),dict(details))
    return TrendEvidence(float(getattr(result,"score",0)),str(getattr(result,"direction","stable")),bool(getattr(result,"score",0)>=0.5),list(getattr(result,"contributing_signals",()) or ()),dict(getattr(result,"details",{}) or {}))

def anomaly_evidence(result: Any)->AnomalyEvidence:
    if result is None: return AnomalyEvidence()
    if isinstance(result, Mapping):
        return AnomalyEvidence(
            float(result.get("probability", 0.0)),
            bool(result.get("is_anomalous", False)),
            list(result.get("contributing_features", []) or []),
            result.get("model_version"),
        )
    return AnomalyEvidence(float(getattr(result,"probability",0.0)),bool(getattr(result,"is_anomalous",False)),list(getattr(result,"contributing_features",[]) or []),getattr(result,"model_version",None))

def risk_evidence(result: Any)->RiskEvidence:
    if result is None: return RiskEvidence()
    if hasattr(result,"to_dict"): result=result.to_dict()
    if isinstance(result,Mapping): return RiskEvidence(float(result.get("risk_score",0)),str(result.get("risk_level",result.get("risk_category","LOW")).upper()),list(result.get("contributing_factors",[]) or []))
    return RiskEvidence(float(getattr(result,"risk_score",0)),str(getattr(result,"risk_level","LOW")).upper(),list(getattr(result,"contributing_factors",[]) or []))

def _case(c, similarity=0.0, experience_confidence=0.0)->SimilarCase:
    d=c.to_dict() if hasattr(c,"to_dict") else (asdict(c) if hasattr(c,"__dataclass_fields__") else dict(c))
    return SimilarCase(str(d.get("case_id","unknown")),float(similarity),dict(d.get("physiological_pattern",{})),float(d.get("anomaly_confidence",0)),dict(d.get("trend_information",{})),d.get("clinician_decision"),d.get("clinician_feedback"),d.get("outcome"),float(experience_confidence))

def experience_from_p3(result: Any)->tuple[list[SimilarCase],float]:
    if result is None: return [],0.0
    cases=getattr(result,"retrieved_cases",[]) or []
    scores=getattr(result,"similarity_scores",[]) or []
    conf=float(getattr(result,"experience_confidence",0.0) or 0.0)
    return [_case(c,scores[i] if i<len(scores) else 0.0,conf) for i,c in enumerate(cases)],conf

def evidence_from_p3(*,evidence_id:str,patient_id:str,window_end:str,current_vitals:Mapping[str,Any],trend_result:Any,anomaly_result:Any,risk_result:Any,patient_context_payload:Mapping[str,Any],experience_result:Any=None,window_start:str|None=None,conflicts_or_missing:list[str]|None=None,fusion_confidence:float=1.0)->ConsolidatedEvidence:
    cases,exp_conf=experience_from_p3(experience_result)
    vitals={k:float(_value(v)) for k,v in current_vitals.items() if _value(v) is not None}
    return ConsolidatedEvidence(SCHEMA_VERSION,evidence_id,patient_id,window_start,window_end,vitals,trend_evidence(trend_result),anomaly_evidence(anomaly_result),risk_evidence(risk_result),patient_context(patient_context_payload),cases,exp_conf,float(fusion_confidence),list(conflicts_or_missing or []))

def evidence_from_fused_evidence(*,evidence_id:str,fused:Any,current_vitals:Mapping[str,Any],patient_context_payload:Mapping[str,Any],trend_result:Any=None,anomaly_result:Any=None,experience_result:Any=None,window_start:str|None=None)->ConsolidatedEvidence:
    return evidence_from_p3(evidence_id=evidence_id,patient_id=str(fused.patient_id),window_end=str(fused.timestamp),current_vitals=current_vitals,trend_result=trend_result,anomaly_result=anomaly_result,risk_result={"risk_score":fused.risk_score,"risk_level":fused.risk_level,"contributing_factors":fused.supporting_evidence.get("risk_factors",[])},patient_context_payload=patient_context_payload,experience_result=experience_result,window_start=window_start,fusion_confidence=fused.final_confidence)

class ExperienceMemoryAdapter:
    """Adapter over DataGen's P3 ExperienceMemory; P4 only sees this interface."""
    def __init__(self,memory:Any): self.memory=memory
    def retrieve_similar_cases(self, evidence:ConsolidatedEvidence, top_k:int=3):
        core={GENERATED_TO_CORE[k]:v for k,v in evidence.current_vitals.items() if k in GENERATED_TO_CORE}
        state=SimpleNamespace(current_vitals=core)
        trend=SimpleNamespace(score=evidence.trend.score)
        anomaly=SimpleNamespace(probability=evidence.anomaly.probability)
        risk=SimpleNamespace(risk_score=evidence.risk.risk_score)
        result=self.memory.retrieve_similar(state,trend,anomaly,risk,top_k=top_k)
        cases=getattr(result,"retrieved_cases",[]) or []
        scores=getattr(result,"similarity_scores",[]) or []
        confidence=float(getattr(result,"experience_confidence",0.0) or 0.0)
        return [_case(c, scores[i] if i < len(scores) else 0.0, confidence) for i,c in enumerate(cases)]
