import json,sys,os
from types import SimpleNamespace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT))
from clinical_reasoning.knowledge_store import TfidfRetriever, PersistentVectorRetriever, load_corpus
from clinical_reasoning.query_builder import build_query
from clinical_reasoning.adapters import evidence_from_p3, ExperienceMemoryAdapter
from clinical_reasoning.reasoning_agent import DeterministicClinicalReasoner, ClinicalReasoningAgent

def corpus(): return ROOT/'knowledge/protocol_corpus.jsonl'

def evidence():
    return evidence_from_p3(evidence_id='e1',patient_id='SYN-0002',window_end='2026-09-18T10:07:00+00:00',current_vitals={'heart_rate':116,'spo2':87,'respiratory_rate':34,'systolic_bp':96,'diastolic_bp':61,'map':72},trend_result={'candidate_deterioration_trend':True,'concurrent_worsening_signals':['heart_rate','spo2','respiratory_rate','systolic_bp']},anomaly_result={'probability':0.8,'is_anomalous':True,'contributing_features':['HR.value','O2Sat.value']},risk_result={'risk_score':0.82,'risk_level':'CRITICAL','contributing_factors':['heart_rate','oxygen_saturation','persistent_trend']},patient_context_payload={'age':72,'sex':'female','relevant_history':['COPD','hypertension'],'current_medications':['antihypertensive'],'recent_labs':{'creatinine_mg_dl':1.73}})

def test_retrieval():
    r=TfidfRetriever.from_jsonl(corpus()); hits=r.retrieve('falling oxygen saturation hypotension respiratory rate',3); assert hits; assert all(h.document.document_id for h in hits)

def test_query():
    q=build_query(evidence()); assert 'hypoxemia' in q and 'hypotension' in q

def test_reasoning_offline():
    out=DeterministicClinicalReasoner(TfidfRetriever.from_jsonl(corpus())).run(evidence()); assert out.patient_id=='SYN-0002'; assert out.citations; assert out.recommended_action

class _TinyEncoder:
    model_id='test-vector-v1'
    def encode(self,texts):
        return __import__('numpy').array([[text.lower().count('blood'), text.lower().count('oxygen'), len(text)] for text in texts], dtype='float32')

def test_dense_vector_index_persists():
    docs=load_corpus(ROOT/'knowledge/protocol_corpus.jsonl')
    index=ROOT/'data'/'derived'/f'p4-test-vectors-{os.getpid()}.sqlite3'
    try:
        retriever=PersistentVectorRetriever(docs,index,backend=_TinyEncoder())
        first=retriever.retrieve('blood pressure',2)
        assert first and index.exists()
        reopened=PersistentVectorRetriever(docs,index,backend=_TinyEncoder())
        assert [hit.document.document_id for hit in first] == [hit.document.document_id for hit in reopened.retrieve('blood pressure',2)]
    finally:
        index.unlink(missing_ok=True)

class _FakeGroq:
    def __init__(self):
        self.chat=SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self.calls=0
    def create(self,**kwargs):
        from clinical_reasoning.knowledge_store import citation_from_chunk
        tool_name=kwargs['tool_choice']['function']['name']
        self.calls+=1
        args={
            'search_clinical_knowledge': {'query':'hypoxemia hypotension falling blood pressure', 'top_k':3},
            'retrieve_similar_cases': {'top_k':3},
            'submit_assessment': {'explanation':'Evidence is concerning; correlate with the retrieved guideline.','contributing_factors':['falling blood pressure'],'recommended_action':'Prompt clinician review and apply local policy.','citations':[],'similar_cases_referenced':[],'confidence':0.7},
        }[tool_name]
        if tool_name=='submit_assessment':
            # Cite an exact result supplied by the local retrieval tool.
            import json as _json
            tool_output=next(m['content'] for m in kwargs['messages'] if m.get('role')=='tool' and m.get('name')=='search_clinical_knowledge')
            row=_json.loads(tool_output)[0]
            args['citations']=[{key:row[key] for key in ('document_id','version','locator','source_url')}]
        call=SimpleNamespace(id=f'call-{self.calls}',function=SimpleNamespace(name=tool_name,arguments=json.dumps(args)))
        message=SimpleNamespace(tool_calls=[call],model_dump=lambda exclude_none=True:{'role':'assistant','tool_calls':[{'id':call.id,'type':'function','function':{'name':tool_name,'arguments':call.function.arguments}}]})
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

def test_groq_tool_flow_and_citations():
    client=_FakeGroq()
    agent=ClinicalReasoningAgent(TfidfRetriever.from_jsonl(ROOT/'knowledge/protocol_corpus.jsonl'),client=client)
    output=agent.run(evidence())
    assert client.calls==3
    assert output.tool_call_count==3
    assert output.citations and output.citations[0].document_id

def test_required_protocol_coverage():
    ids={doc.document_id for doc in load_corpus(corpus())}
    assert {'sepsis3-definition-2016','sepsis3-qsofa-2016','news2-rcp-2017','nice-ng51-2016','nice-ng253-adults-2025','nice-ng254-children-2025','nice-ng255-pregnancy-2025'} <= ids

def test_experience_memory_adapter():
    from experience import ExperienceMemory, HistoricalCase
    memory=ExperienceMemory()
    memory.store_case(HistoricalCase(
        case_id='prior-1',patient_id='SYN-OLD',timestamp='2026-01-01',
        physiological_pattern={'HR':116,'O2Sat':87,'Resp':34,'SBP':96,'DBP':61},
        trend_information={'score':0.8},anomaly_confidence=0.8,risk_score=0.82,
        clinician_decision='review',clinician_feedback='appropriate',outcome='observed',
    ))
    result=ExperienceMemoryAdapter(memory).retrieve_similar_cases(evidence())
    assert result and result[0].case_id=='prior-1'
