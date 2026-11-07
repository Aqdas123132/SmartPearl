"""Weak vs final prompt on the same five questions (needs a real LLM: LLM_PROVIDER=openai or ollama).
python -m evaluation.prompt_experiment"""
from __future__ import annotations

import json
from pathlib import Path

from src.chains.rag_chain import SupportAgent
from src.config import get_settings
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import load_or_build

HERE = Path(__file__).parent
IDS = ["T-12", "T-13", "T-17", "T-18", "T-21"]  # the cases that tempt a weak prompt to hallucinate or comply


def main() -> None:
    if get_settings().llm_provider == "extractive":
        raise SystemExit("The prompt experiment compares LLM behaviour. Set LLM_PROVIDER=openai (and OPENAI_API_KEY) or ollama in .env first.")
    cases = {c["id"]: c for c in json.loads((HERE / "test_questions.json").read_text())}
    r = Retriever(load_or_build())
    out = []
    for cid in IDS:
        row = {"id": cid, "question": cases[cid]["turns"][-1]}
        for version in ("weak", "final"):
            # the injection guard is part of the final pipeline; for a fair prompt-only comparison it is only active for 'final'
            agent = SupportAgent(r, prompt_version=version)
            if version == "weak":
                agent.s = agent.s.model_copy(update={"score_threshold": -1.0})  # disable the gate so the PROMPT alone is tested
                agent._guard = lambda x: {"blocked": False, "guess": False}
                agent.chain = agent._build_chain()
            resp = agent.ask(cases[cid]["turns"][-1])
            row[version] = {"answer": resp.answer, "answered": resp.answered, "citations": [c.source for c in resp.sources]}
        out.append(row)
    (HERE / "results" / "prompt_experiment.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for row in out:
        print(f"\n{row['id']}: {row['question']}\n  WEAK : {row['weak']['answer'][:300]}\n  FINAL: {row['final']['answer'][:300]}")


if __name__ == "__main__":
    main()
