from __future__ import annotations
from typing import Protocol
from .schema import ConsolidatedEvidence, SimilarCase

class ExperienceMemoryClient(Protocol):
    def retrieve_similar_cases(self,evidence:ConsolidatedEvidence,top_k:int=3)->list[SimilarCase]: ...

class InMemoryExperienceClient:
    def __init__(self,cases=None): self.cases=list(cases or [])
    def retrieve_similar_cases(self,evidence,top_k=3): return sorted(self.cases,key=lambda c:c.similarity,reverse=True)[:top_k]
