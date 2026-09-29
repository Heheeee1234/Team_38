"""Build or refresh the local P4 dense-vector index from the versioned corpus."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinical_reasoning.knowledge_store import PersistentVectorRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "knowledge" / "protocol_corpus.jsonl")
    parser.add_argument("--index", type=Path, default=ROOT / "data" / "derived" / "p4_vector_index.sqlite3")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()
    retriever = PersistentVectorRetriever.from_jsonl(
        args.corpus, args.index, model_id=args.model,
    )
    print(f"Indexed {len(retriever.chunks)} chunks from {args.corpus}")
    print(f"Local vector index: {args.index}")


if __name__ == "__main__":
    main()
