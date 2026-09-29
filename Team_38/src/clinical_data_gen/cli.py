from __future__ import annotations
import argparse, json, time
from datetime import datetime, timezone
from pathlib import Path
from .generator import ClinicalDataGenerator, sha256, write_jsonl
from .fit_physionet import write_fit
from .fidelity import report as fidelity_report
from .fit_nhanes import write_fit as write_nhanes_fit

def rows(path: Path) -> list[dict]: return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]
def validate_directory(folder: Path) -> dict:
    vitals,contexts,labels=rows(folder/"vitals_observed.jsonl"),rows(folder/"patient_context.jsonl"),rows(folder/"evaluation_labels.jsonl")
    assert vitals and contexts and len(vitals)==len(labels), "missing records or labels"
    assert all("scenario" not in e["payload"] and "latent" not in e["payload"] for e in vitals), "label leakage in live event"
    assert all(70<=e["payload"]["vitals"]["spo2"]<=100 and e["payload"]["vitals"]["systolic_bp"]>e["payload"]["vitals"]["diastolic_bp"] for e in vitals), "physiology invariant failure"
    return {"status":"pass","vital_events":len(vitals),"patient_contexts":len(contexts),"episodes":len(rows(folder/"ground_truth_episodes.jsonl")),"leakage_check":"pass","physiology_bounds_check":"pass"}
def generate(args: argparse.Namespace) -> None:
    config=json.loads(Path(args.config).read_text()); out=Path(args.output); out.mkdir(parents=True,exist_ok=True); fitted=json.loads(Path(args.params).read_text()) if args.params else None; records=ClinicalDataGenerator(args.seed,config,fitted).create(args.patients,args.minutes)
    for name,value in records.items(): write_jsonl(out/f"{name}.jsonl",value)
    (out/"knowledge_corpus.jsonl").write_text(Path(args.knowledge).read_text(encoding="utf-8"),encoding="utf-8")
    quality=validate_directory(out)
    if args.reference: quality["fidelity"]=fidelity_report(Path(args.reference),out,Path(args.params),args.fidelity_max_files)
    (out/"data_quality_report.json").write_text(json.dumps(quality,indent=2))
    manifest={"schema_version":config["schema_version"],"seed":args.seed,"patients":args.patients,"minutes":args.minutes,"run_id":records["vitals_observed"][0]["run_id"],"config_sha256":sha256(Path(args.config)),"params_sha256":sha256(Path(args.params)) if args.params else None,"knowledge_corpus_sha256":sha256(Path(args.knowledge)),"files":{p.name:sha256(p) for p in out.glob("*.jsonl")}}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)); print(json.dumps(quality,indent=2))
def replay(args: argparse.Namespace) -> None:
    if args.speed <= 0:
        raise ValueError("--speed must be greater than zero")
    contexts = rows(Path(args.input) / "patient_context.jsonl")
    vitals = rows(Path(args.input) / "vitals_observed.jsonl")
    vitals.sort(key=lambda event: _event_time(event["event_time"]))
    producer=None
    if args.brokers:
        try:
            from confluent_kafka import Producer; producer=Producer({"bootstrap.servers":args.brokers})
        except ImportError as e: raise SystemExit("Install optional dependency: pip install confluent-kafka") from e
    def emit(topic,e):
        if producer: producer.produce(topic,key=e["patient_id"],value=json.dumps(e)); producer.poll(0)
        else: print(json.dumps({"topic":topic,"key":e["patient_id"],"value":e}))
    for e in contexts: emit("patient.context",e)
    previous_time = None
    for e in vitals:
        current_time = _event_time(e["event_time"])
        if previous_time is not None:
            delay = (current_time - previous_time).total_seconds()
            if delay < 0:
                raise ValueError("Vital events are not ordered by event_time")
            if delay:
                time.sleep(delay / args.speed)
        emit("vitals.raw",e)
        previous_time = current_time
    if producer: producer.flush()

def _event_time(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"Invalid event_time during replay: {value!r}") from exc
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
def main() -> None:
    p=argparse.ArgumentParser(); s=p.add_subparsers(required=True)
    g=s.add_parser("generate"); g.add_argument("--output",required=True); g.add_argument("--patients",type=int,default=60); g.add_argument("--minutes",type=int,default=180); g.add_argument("--seed",type=int,default=20260918); g.add_argument("--config",default="config/generator_config.json"); g.add_argument("--params",default="data/derived/physionet_2019_params.json"); g.add_argument("--knowledge",default="knowledge/protocol_corpus.jsonl"); g.add_argument("--reference"); g.add_argument("--fidelity-max-files",type=int,default=2500); g.set_defaults(func=generate)
    f=s.add_parser("fit-physionet"); f.add_argument("--input",required=True); f.add_argument("--output",default="data/derived/physionet_2019_params.json"); f.add_argument("--max-files",type=int); f.set_defaults(func=lambda a:print(json.dumps(write_fit(Path(a.input),Path(a.output),a.max_files)["source"],indent=2)))
    n=s.add_parser("fit-nhanes-baselines"); n.add_argument("--bp",required=True); n.add_argument("--demographics",required=True); n.add_argument("--output",default="data/derived/nhanes_2017_2018_baselines.json"); n.set_defaults(func=lambda a:print(json.dumps(write_nhanes_fit(Path(a.bp),Path(a.demographics),Path(a.output))["source"],indent=2)))
    v=s.add_parser("validate"); v.add_argument("--input",required=True); v.set_defaults(func=lambda a:print(json.dumps(validate_directory(Path(a.input)),indent=2)))
    r=s.add_parser("replay"); r.add_argument("--input",required=True); r.add_argument("--brokers"); r.add_argument("--speed",type=float,default=60); r.set_defaults(func=replay)
    a=p.parse_args(); a.func(a)
if __name__=="__main__": main()
