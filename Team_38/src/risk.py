from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from clinical_data_gen.state.patient_state import PatientState
from clinical_data_gen.trend import TrendResult

RISK_LEVELS = ("LOW", "MODERATE", "HIGH", "CRITICAL")


@dataclass(frozen=True)
class RiskThresholds:
    low: float = 0.2
    moderate: float = 0.45
    high: float = 0.7

    def level_for(self, score: float) -> str:
        if score >= self.high:
            return "CRITICAL"
        if score >= self.moderate:
            return "HIGH"
        if score >= self.low:
            return "MODERATE"
        return "LOW"


@dataclass(frozen=True)
class RiskScoringConfig:
    """Deterministic, configuration-driven risk scoring strategy."""

    thresholds: RiskThresholds = field(default_factory=RiskThresholds)
    hr_center: float = 80.0
    hr_scale: float = 30.0
    spo2_center: float = 97.0
    spo2_scale: float = 8.0
    rr_center: float = 18.0
    rr_scale: float = 10.0
    sbp_center: float = 120.0
    sbp_scale: float = 30.0
    dbp_center: float = 70.0
    dbp_scale: float = 20.0
    anomaly_weight: float = 0.45
    trend_weight: float = 0.25
    physiology_weight: float = 0.30


@dataclass(frozen=True)
class RiskResult:
    risk_score: float
    risk_level: str
    contributing_factors: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "contributing_factors": list(self.contributing_factors),
        }


class RiskScoringStrategy:
    """Abstract scoring strategy; can be swapped without changing the service API."""

    def score(
        self,
        patient_state: PatientState | Mapping[str, Any],
        trend_result: TrendResult | Mapping[str, Any] | None,
        anomaly_result: Any,
        config: RiskScoringConfig,
    ) -> float:
        raise NotImplementedError


class WeightedPhysiologyRiskStrategy(RiskScoringStrategy):
    """A transparent, configurable, non-LLM risk strategy."""

    def score(
        self,
        patient_state: PatientState | Mapping[str, Any],
        trend_result: TrendResult | Mapping[str, Any] | None,
        anomaly_result: Any,
        config: RiskScoringConfig,
    ) -> float:
        physiology_score = self._physiology_score(
            patient_state,
            config,
        )
        trend_score = self._trend_score(trend_result)
        anomaly_score = self._anomaly_score(anomaly_result)

        total = (
            config.physiology_weight * physiology_score
            + config.trend_weight * trend_score
            + config.anomaly_weight * anomaly_score
        )
        return max(0.0, min(1.0, total))

    @staticmethod
    def _physiology_score(
        patient_state: PatientState | Mapping[str, Any],
        config: RiskScoringConfig,
    ) -> float:
        if patient_state is None:
            return 0.0

        vitals = getattr(patient_state, "current_vitals", {})
        if isinstance(vitals, Mapping):
            current_vitals = vitals
        else:
            current_vitals = {}

        weight_total = 0.0
        score_total = 0.0

        for vital_name, weight in (
            ("HR", 1.0),
            ("O2Sat", 1.5),
            ("Resp", 1.2),
            ("SBP", 1.0),
            ("DBP", 1.0),
        ):
            value = _extract_vital(current_vitals, vital_name)
            if value is None:
                continue
            deviation = _normalized_deviation(value, vital_name, config)
            score_total += deviation * weight
            weight_total += weight

        if weight_total == 0.0:
            return 0.0
        return min(1.0, score_total / weight_total)

    @staticmethod
    def _trend_score(
        trend_result: TrendResult | Mapping[str, Any] | None,
    ) -> float:
        if trend_result is None:
            return 0.0
        if isinstance(trend_result, Mapping):
            score = trend_result.get("score", 0.0)
            if score is None:
                return 0.0
            return max(0.0, min(1.0, float(score)))
        return max(0.0, min(1.0, float(getattr(trend_result, "score", 0.0))))

    @staticmethod
    def _anomaly_score(
        anomaly_result: Any,
    ) -> float:
        if anomaly_result is None:
            return 0.0
        if isinstance(anomaly_result, Mapping):
            score = anomaly_result.get("probability", anomaly_result.get("score", 0.0))
            if score is None:
                return 0.0
            return max(0.0, min(1.0, float(score)))
        probability = getattr(anomaly_result, "probability", None)
        if probability is None:
            probability = getattr(anomaly_result, "score", 0.0)
        return max(0.0, min(1.0, float(probability)))


class RiskAssessor:
    """Deterministic risk assessment for a single patient observation."""

    def __init__(
        self,
        *,
        strategy: RiskScoringStrategy | None = None,
        config: RiskScoringConfig | None = None,
    ) -> None:
        self.strategy = strategy or WeightedPhysiologyRiskStrategy()
        self.config = config or RiskScoringConfig()

    def assess(
        self,
        patient_state: PatientState | Mapping[str, Any],
        trend_result: TrendResult | Mapping[str, Any] | None,
        anomaly_result: Any,
    ) -> RiskResult:
        score = self.strategy.score(
            patient_state,
            trend_result,
            anomaly_result,
            self.config,
        )
        risk_level = self.config.thresholds.level_for(score)
        contributing_factors = self._contributing_factors(
            patient_state,
            trend_result,
            anomaly_result,
        )
        return RiskResult(
            risk_score=float(score),
            risk_level=risk_level,
            contributing_factors=contributing_factors,
        )

    def _contributing_factors(
        self,
        patient_state: PatientState | Mapping[str, Any],
        trend_result: TrendResult | Mapping[str, Any] | None,
        anomaly_result: Any,
    ) -> list[str]:
        factors: list[str] = []
        if patient_state is not None:
            vitals = getattr(patient_state, "current_vitals", {})
            if isinstance(vitals, Mapping):
                current_vitals = vitals
            else:
                current_vitals = {}

            for vital_name, label in (
                ("HR", "heart_rate"),
                ("O2Sat", "oxygen_saturation"),
                ("Resp", "respiratory_rate"),
                ("SBP", "systolic_bp"),
                ("DBP", "diastolic_bp"),
            ):
                value = _extract_vital(current_vitals, vital_name)
                if value is None:
                    continue
                deviation = _normalized_deviation(value, vital_name, self.config)
                if deviation >= 0.5:
                    factors.append(label)

        trend_score = WeightedPhysiologyRiskStrategy._trend_score(trend_result)
        if trend_score >= 0.5:
            factors.append("persistent_trend")

        anomaly_score = WeightedPhysiologyRiskStrategy._anomaly_score(anomaly_result)
        if anomaly_score >= 0.5:
            factors.append("anomaly_probability")

        if not factors:
            factors.append("stable_observation")

        return factors[:5]


def _extract_vital(current_vitals: Mapping[str, Any], vital_name: str) -> float | None:
    if vital_name in current_vitals:
        value = _observation_value(current_vitals.get(vital_name))
        if value is not None:
            return float(value)

    alias_map = {
        "O2Sat": ("SpO2", "spo2", "O2Sat"),
        "Resp": ("RR", "resp", "Resp"),
        "SBP": ("SystolicBP", "sys_bp", "SBP"),
        "DBP": ("DiastolicBP", "dia_bp", "DBP"),
        "HR": ("heart_rate", "hr", "HR"),
    }
    for alias in alias_map.get(vital_name, (vital_name,)):
        if alias in current_vitals:
            value = _observation_value(current_vitals.get(alias))
            if value is not None:
                return float(value)
    return None


def _observation_value(observation: Any) -> Any:
    for _ in range(3):
        if isinstance(observation, Mapping):
            observation = observation.get("value")
        elif hasattr(observation, "value"):
            observation = observation.value
        else:
            return observation
    return observation


def _normalized_deviation(
    value: float, vital_name: str, config: RiskScoringConfig
) -> float:
    if vital_name == "HR":
        center, scale = config.hr_center, config.hr_scale
    elif vital_name == "O2Sat":
        center, scale = config.spo2_center, config.spo2_scale
    elif vital_name == "Resp":
        center, scale = config.rr_center, config.rr_scale
    elif vital_name == "SBP":
        center, scale = config.sbp_center, config.sbp_scale
    elif vital_name == "DBP":
        center, scale = config.dbp_center, config.dbp_scale
    else:
        return 0.0

    if scale <= 0:
        return 0.0

    if vital_name == "O2Sat":
        deviation = max(0.0, (center - value) / scale)
    else:
        deviation = min(1.0, max(0.0, abs(value - center) / scale))
    return deviation
