from __future__ import annotations
from dataclasses import replace
from pathlib import Path
from .knowledge_store import PersistentVectorRetriever, TfidfRetriever
from .reasoning_agent import ClinicalReasoningAgent, DeterministicClinicalReasoner

class P4Service:
    """Thin service boundary: P3 evidence in -> P4 reasoning out."""
    def __init__(self, corpus: str|Path, *, live_llm=False, experience_client=None,
                 retrieval_backend="dense", index_path="data/derived/p4_vector_index.sqlite3",
                 embedding_model="sentence-transformers/all-MiniLM-L6-v2", model=None):
        if retrieval_backend == "dense":
            self.retriever = PersistentVectorRetriever.from_jsonl(
                corpus, index_path, model_id=embedding_model
            )
        elif retrieval_backend == "tfidf":
            self.retriever = TfidfRetriever.from_jsonl(corpus)
        else:
            raise ValueError("retrieval_backend must be 'dense' or 'tfidf'")
        self.experience_client=experience_client
        self.live_llm=live_llm
        self.agent=ClinicalReasoningAgent(
            self.retriever, experience_client=experience_client, model=model or None
        ) if live_llm else DeterministicClinicalReasoner(self.retriever)
    def reason(self,evidence):
        """Run RAG reasoning, adding live P3 memory results when configured."""
        if self.experience_client is not None:
            cases = self.experience_client.retrieve_similar_cases(evidence)
            evidence = replace(evidence, experience_cases=cases)
            if cases:
                evidence = replace(
                    evidence,
                    experience_confidence=max(
                        evidence.experience_confidence,
                        max(float(getattr(case, "experience_confidence", 0.0)) for case in cases),
                    ),
                )
        return self.agent.run(evidence)
