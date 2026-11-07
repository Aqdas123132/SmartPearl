import os
import tempfile
from pathlib import Path

_tmp = Path(tempfile.mkdtemp(prefix="pearlz_test_"))
os.environ.update({"VECTOR_STORE_PATH": str(_tmp / "vs"), "EMBEDDING_CACHE_PATH": str(_tmp / "emb.sqlite"), "DATABASE_URL": f"sqlite:///{_tmp / 'app.db'}",
                   "LOG_DIR": str(_tmp / "logs"), "LLM_PROVIDER": "extractive", "EMBEDDING_PROVIDER": "hash", "SECRET_KEY": "test-secret-key-0123456789"})

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda


@pytest.fixture(scope="session")
def retriever():
    from src.retrieval.retriever import Retriever
    from src.retrieval.vector_store import load_or_build
    return Retriever(load_or_build())


class StubLLM(RunnableLambda):
    """Stands in for a chat model: `answer_fn(prompt_value)` returns an LLMAnswer or raises; used as the condenser too."""

    def __init__(self, answer_fn, condense_fn=lambda pv: "condensed"):
        super().__init__(lambda pv: AIMessage(content=condense_fn(pv)))
        object.__setattr__(self, "answer_fn", answer_fn)

    def with_structured_output(self, schema, include_raw=False):
        def run(pv):
            res = self.answer_fn(pv)
            return {"raw": AIMessage(content="", usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}), "parsed": res, "parsing_error": None if res else ValueError("bad")}
        return RunnableLambda(run)


@pytest.fixture
def make_agent(retriever):
    from src.chains.rag_chain import SupportAgent

    def _make(answer_fn):
        agent = SupportAgent(retriever)
        agent.llm = StubLLM(answer_fn)
        agent.chain = agent._build_chain()
        return agent
    return _make
