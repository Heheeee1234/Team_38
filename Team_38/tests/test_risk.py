from clinical_data_gen import (
    ConfidenceFusion,
    EvidenceFusion,
    ExperienceMemory,
    HistoricalCase,
    PatientState,
    RiskAssessor,
    RiskScoringConfig,
    TrendResult,
)


def _state() -> PatientState:
    state = PatientState(patient_id="p-1")
    state.set_vital("HR", 110.0)
    state.set_vital("O2Sat", 89.0)
    state.set_vital("Resp", 28.0)
    state.set_vital("SBP", 88.0)
    state.set_vital("DBP", 55.0)
    return state


def test_risk_assessor_uses_configurable_scoring_and_levels():
    state = _state()
    trend = TrendResult(patient_id="p-1", score=0.8, direction="worsening")
    anomaly = {"probability": 0.9}

    assessor = RiskAssessor(config=RiskScoringConfig())
    result = assessor.assess(state, trend, anomaly)

    assert 0.0 <= result.risk_score <= 1.0
    assert result.risk_level in {"LOW", "MODERATE", "HIGH", "CRITICAL"}
    assert result.contributing_factors


def test_confidence_fusion_validates_bounds_and_alpha():
    fusion = ConfidenceFusion(alpha=0.7)

    assert 0.0 <= fusion.combine(0.9, 0.2) <= 1.0
    assert fusion.combine(1.0, 1.0) == 1.0

    try:
        fusion.combine(1.1, 0.5)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass

    try:
        fusion.combine(0.5, 0.5, alpha=2.0)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_experience_memory_retrieves_similar_cases_and_computes_confidence():
    memory = ExperienceMemory()
    case = HistoricalCase(
        case_id="case-1",
        patient_id="p-1",
        timestamp=1,
        physiological_pattern={
            "HR": 110.0,
            "O2Sat": 89.0,
            "Resp": 28.0,
            "SBP": 88.0,
            "DBP": 55.0,
        },
        trend_information={"score": 0.8},
        anomaly_confidence=0.9,
        risk_score=0.8,
        clinician_decision="escalate",
        clinician_feedback="Escalated for review.",
        outcome="improved",
    )
    memory.store_case(case)

    result = memory.retrieve_similar(
        _state(),
        TrendResult(patient_id="p-1", score=0.8),
        {"probability": 0.9},
        {"risk_score": 0.8},
        top_k=3,
    )

    assert result.experience_confidence >= 0.0
    assert result.experience_confidence <= 1.0
    assert len(result.retrieved_cases) >= 1
    assert result.retrieved_cases[0].case_id == "case-1"


def test_evidence_fusion_aggregates_inputs():
    state = _state()
    trend = TrendResult(patient_id="p-1", score=0.8)
    anomaly = {"probability": 0.9}
    risk = RiskAssessor().assess(state, trend, anomaly)
    memory = ExperienceMemory()
    memory.store_case(
        HistoricalCase(
            case_id="case-a",
            patient_id="p-1",
            timestamp=2,
            physiological_pattern={
                "HR": 108.0,
                "O2Sat": 90.0,
                "Resp": 27.0,
                "SBP": 90.0,
                "DBP": 58.0,
            },
            trend_information={"score": 0.7},
            anomaly_confidence=0.85,
            risk_score=0.75,
            clinician_decision="observe",
            clinician_feedback="Signs worsened but stable.",
            outcome="improved",
        )
    )
    experience = memory.retrieve_similar(state, trend, anomaly, risk, top_k=3)
    fused = EvidenceFusion().fuse(state, trend, anomaly, risk, experience)

    assert 0.0 <= fused.final_confidence <= 1.0
    assert fused.patient_id == "p-1"
    assert fused.risk_level in {"LOW", "MODERATE", "HIGH", "CRITICAL"}
