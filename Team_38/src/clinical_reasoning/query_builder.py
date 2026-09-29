from __future__ import annotations
from .schema import ConsolidatedEvidence

_RULES=(
("heart_rate",">",100,"tachycardia rising heart rate"),
("heart_rate","<",50,"bradycardia"),
("spo2","<",92,"hypoxemia falling oxygen saturation"),
("respiratory_rate",">",22,"tachypnea rising respiratory rate"),
("respiratory_rate","<",8,"respiratory depression"),
("systolic_bp","<",100,"hypotension falling blood pressure"),
("map","<",65,"low mean arterial pressure organ hypoperfusion"),
("diastolic_bp","<",60,"low diastolic blood pressure"),
)

def build_query(e: ConsolidatedEvidence)->str:
    terms=[]
    for vital,op,threshold,phrase in _RULES:
        value=e.current_vitals.get(vital)
        if value is None: continue
        if (op==">" and value>threshold) or (op=="<" and value<threshold): terms.append(phrase)
    if e.trend.persistent_deterioration: terms.append("sustained multi-reading deterioration trend")
    if e.trend.contributing_signals: terms.extend(e.trend.contributing_signals)
    if e.anomaly.contributing_features: terms.append("anomaly " + " ".join(e.anomaly.contributing_features))
    if e.risk.risk_level in {"HIGH","CRITICAL"}: terms.append("high risk organ dysfunction suspected infection sepsis")
    if e.patient_context.relevant_history: terms.extend(e.patient_context.relevant_history[:3])
    return " ".join(terms) or "routine vital sign monitoring stable patient"
