from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import asdict
from typing import Any

from .knowledge_store import citation_from_chunk
from .query_builder import build_query
from .schema import *

DEFAULT_MODEL = "openai/gpt-oss-20b"


class ReasoningAgentError(RuntimeError):
    pass


SYSTEM_PROMPT = """You are a clinical decision-support reasoning component for a synthetic
clinical deterioration demonstration. Do not diagnose, prescribe, or issue treatment
orders. Explain only the supplied P3-fused evidence. Treat missing/conflicting data as
uncertainty. Use the clinical knowledge and similar-case tool results as context.
Ground clinical claims in retrieved guideline passages and cite only their exact
document_id, version, and locator. Similar cases are contextual evidence, not guidelines.
Recommended actions must be cautious prompts for clinician review and must defer to
current local policy and clinician judgement. Never infer that a low/default risk means
the patient is safe. Complete the forced tool workflow and submit one structured
assessment."""


def _tool(name: str, description: str, properties: dict, required: list[str] | None = None):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required or [],
                "additionalProperties": False,
            },
        },
    }


TOOLS = [
    _tool("search_clinical_knowledge", "Search the local versioned clinical protocol corpus.", {
        "query": {"type": "string"}, "top_k": {"type": "integer", "minimum": 1, "maximum": 5},
    }, ["query"]),
    _tool("retrieve_similar_cases", "Retrieve similar experience-memory cases and outcomes.", {
        "top_k": {"type": "integer", "minimum": 1, "maximum": 5},
    }),
    _tool("submit_assessment", "Submit the structured P4 reasoning result after both retrieval tools.", {
        "explanation": {"type": "string"},
        "contributing_factors": {"type": "array", "items": {"type": "string"}},
        "recommended_action": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "object", "properties": {
            "document_id": {"type": "string"}, "version": {"type": "string"},
            "locator": {"type": "string"}, "source_url": {"type": "string"},
        }, "required": ["document_id", "version", "locator"], "additionalProperties": False}},
        "similar_cases_referenced": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    }, ["explanation", "contributing_factors", "recommended_action", "citations", "confidence"]),
]


class ClinicalReasoningAgent:
    """Groq tool-calling agent; retrieval and tool execution stay on this machine."""

    def __init__(self, retriever, experience_client: Any = None, client: Any = None,
                 model: str = DEFAULT_MODEL, max_tool_iterations: int = 6):
        self.retriever = retriever
        self.experience_client = experience_client
        self.client = client
        self.model = model or DEFAULT_MODEL
        self.max_tool_iterations = max(3, max_tool_iterations)

    def _client(self):
        if self.client is not None:
            return self.client
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise ReasoningAgentError(
                "GROQ_API_KEY is not set. Set it in your terminal environment and retry."
            )
        try:
            from groq import Groq
        except ImportError as exc:
            raise ReasoningAgentError(
                "Groq client is missing. Install the P4 extra: pip install -e \".[p4-groq]\"."
            ) from exc
        self.client = Groq(api_key=key)
        return self.client

    @staticmethod
    def _dump_message(message: Any) -> dict:
        if isinstance(message, dict):
            return message
        return message.model_dump(exclude_none=True)

    def _execute(self, name: str, arguments: dict, evidence: ConsolidatedEvidence) -> list[dict]:
        top_k = max(1, min(5, int(arguments.get("top_k", 3))))
        if name == "search_clinical_knowledge":
            query = str(arguments.get("query") or build_query(evidence))
            return [asdict(citation_from_chunk(hit))
                    for hit in self.retriever.retrieve(query, top_k)]
        if name == "retrieve_similar_cases":
            if self.experience_client is not None:
                result = self.experience_client.retrieve_similar_cases(evidence, top_k)
                if isinstance(result, list):
                    return [asdict(case) if hasattr(case, "__dataclass_fields__") else case
                            for case in result]
                return [case.to_dict() if hasattr(case, "to_dict") else asdict(case)
                        for case in (getattr(result, "retrieved_cases", []) or [])[:top_k]]
            return [asdict(case) for case in evidence.experience_cases[:top_k]]
        raise ReasoningAgentError(f"Unknown tool: {name}")

    def run(self, evidence: ConsolidatedEvidence) -> ReasoningOutput:
        client = self._client()
        user = (
            "P3 consolidated evidence JSON:\n"
            + json.dumps(evidence.to_dict(), indent=2, default=str)
            + "\n\nSuggested clinical retrieval query: " + build_query(evidence)
        )
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user}]
        required_sequence = ["search_clinical_knowledge", "retrieve_similar_cases",
                             "submit_assessment"]
        retrieved: list[dict] = []
        retrieved_case_ids: set[str] = set()
        tool_calls = 0

        for expected_name in required_sequence:
            tool_choice = {"type": "function", "function": {"name": expected_name}}
            response = client.chat.completions.create(
                model=self.model, messages=messages, tools=TOOLS,
                tool_choice=tool_choice, max_completion_tokens=1600,
            )
            message = response.choices[0].message
            calls = getattr(message, "tool_calls", None) or []
            if not calls:
                raise ReasoningAgentError(f"Groq did not call required tool {expected_name!r}.")
            messages.append(self._dump_message(message))
            final_submission = None
            for call in calls:
                tool_calls += 1
                name = call.function.name
                if name != expected_name:
                    raise ReasoningAgentError(f"Expected {expected_name!r}; model called {name!r}.")
                try:
                    arguments = json.loads(call.function.arguments or "{}")
                except (TypeError, json.JSONDecodeError) as exc:
                    raise ReasoningAgentError(f"Invalid JSON arguments for {name}.") from exc
                if name == "submit_assessment":
                    final_submission = arguments
                    result = {"received": True}
                else:
                    result = self._execute(name, arguments, evidence)
                    if name == "search_clinical_knowledge":
                        retrieved.extend(result)
                    else:
                        retrieved_case_ids.update(str(row.get("case_id")) for row in result
                                                   if row.get("case_id"))
                messages.append({
                    "role": "tool", "tool_call_id": call.id, "name": name,
                    "content": json.dumps(result, default=str),
                })
            if final_submission is not None:
                return self._finalize(final_submission, evidence, tool_calls,
                                      retrieved, retrieved_case_ids)
        raise ReasoningAgentError("Groq reasoning reached the end without submitting an assessment.")

    def _finalize(self, submission: dict, evidence: ConsolidatedEvidence,
                  calls: int, retrieved: list[dict], case_ids: set[str]) -> ReasoningOutput:
        valid = {(row["document_id"], row["version"], row["locator"]): row
                 for row in retrieved}
        citations = []
        for citation in submission.get("citations", []):
            key = (citation.get("document_id"), citation.get("version"),
                   citation.get("locator"))
            if key not in valid:
                raise ReasoningAgentError(f"Model returned an ungrounded citation: {key}")
            source = valid[key]
            citations.append(Citation(source["document_id"], source["version"],
                                      source["locator"], source.get("source_url", ""),
                                      source.get("score", 0.0), source.get("chunk_text", "")))
        referenced_cases = list(submission.get("similar_cases_referenced", []))
        unknown = set(referenced_cases) - case_ids
        if unknown:
            raise ReasoningAgentError(f"Model referenced cases not returned by memory: {sorted(unknown)}")
        return ReasoningOutput(
            SCHEMA_VERSION, evidence.patient_id, evidence.evidence_id,
            dt.datetime.now(dt.timezone.utc).isoformat(),
            str(submission["explanation"]), list(submission["contributing_factors"]),
            str(submission["recommended_action"]), citations, referenced_cases,
            max(0.0, min(1.0, float(submission["confidence"]))), self.model, calls,
        )


class DeterministicClinicalReasoner:
    """No-key fallback. Its lexical retrieval and fixed rules work fully offline."""

    def __init__(self, retriever, top_k: int = 3):
        self.retriever = retriever
        self.top_k = top_k

    def run(self, evidence: ConsolidatedEvidence) -> ReasoningOutput:
        query = build_query(evidence)
        hits = self.retriever.retrieve(query, self.top_k)
        factors = []
        for key, value in evidence.current_vitals.items():
            if key == "spo2" and value < 92:
                factors.append(f"SpO2 {value:g}%")
            elif key == "heart_rate" and value > 100:
                factors.append(f"heart rate {value:g}")
            elif key == "respiratory_rate" and value > 22:
                factors.append(f"respiratory rate {value:g}")
            elif key == "systolic_bp" and value < 100:
                factors.append(f"systolic blood pressure {value:g}")
            elif key == "map" and value < 65:
                factors.append(f"MAP {value:g}")
        if evidence.trend.persistent_deterioration:
            factors.append("persistent multi-reading deterioration trend")
        if evidence.anomaly.probability >= 0.5:
            factors.append(f"anomaly probability {evidence.anomaly.probability:.2f}")
        factors.extend(evidence.risk.contributing_factors[:3])
        missing_upstream = any(
            marker in item.lower()
            for item in evidence.conflicts_or_missing
            for marker in ("risk", "trend", "anomaly", "fusion", "p3")
        )
        if missing_upstream:
            action = ("No clinical action recommendation is generated because upstream trend, "
                      "anomaly, risk, or fused evidence is missing. Supply consolidated P3 "
                      "evidence and follow local clinical policy.")
        elif evidence.risk.risk_level in {"HIGH", "CRITICAL"} or evidence.trend.persistent_deterioration:
            action = ("Prompt clinician review of the evolving multi-parameter deterioration "
                      "and correlate with current clinical context; this is not a diagnosis "
                      "or treatment order.")
        elif evidence.risk.risk_level == "MODERATE":
            action = "Continue close observation and clinician review if concerning evidence persists."
        else:
            action = "Continue routine monitoring and reassess if the patient trajectory changes."
        explanation = ("The supplied evidence indicates " +
                       ("; ".join(factors) if factors else "no dominant deterioration signal") +
                       ". Retrieved protocol passages provide context, not a diagnosis.")
        return ReasoningOutput(
            SCHEMA_VERSION, evidence.patient_id, evidence.evidence_id,
            dt.datetime.now(dt.timezone.utc).isoformat(), explanation, factors, action,
            [citation_from_chunk(hit) for hit in hits],
            [case.case_id for case in evidence.experience_cases[:self.top_k]],
            min(1.0, max(0.0, 0.4 * evidence.fusion_confidence +
                         0.3 * evidence.anomaly.probability +
                         0.3 * evidence.experience_confidence)),
            "deterministic-p4", 0,
        )
