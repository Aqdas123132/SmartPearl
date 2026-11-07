"""The composed support chain (LCEL): guard -> condense -> retrieve -> context -> [refusal | generate] -> validate citations."""
from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableBranch, RunnableLambda, RunnablePassthrough
from pydantic import ValidationError

from src.chains import prompts
from src.chains.llm import auth_errors, get_chat_model, transient_errors
from src.chains.memory import heuristic_condense, history_text, to_messages
from src.chains.schemas import Citation, Confidence, LLMAnswer, SupportResponse
from src.config import Settings, get_settings
from src.retrieval.embeddings import STOP, _stem
from src.retrieval.retriever import Hit, RetrievalResult, Retriever
from src.utils.logging_setup import log_turn

log = logging.getLogger(__name__)

INJECTION = re.compile(r"(ignore|disregard|forget)\s+(all\s+|any\s+|your\s+)?(previous|prior|above|earlier)?\s*(instructions|rules|prompt)|developer mode|"
                       r"(print|reveal|show|repeat)\s+(me\s+)?(your|the)\s+(system\s+)?(prompt|instructions)|you are now\b|jailbreak|"
                       r"approve\s+(a|my|the|any)?\s*(full\s+)?refund", re.I)
GUESS = re.compile(r"best guess|just guess|you don'?t need sources|without sources|make (it|something) up", re.I)


class OutputValidationError(RuntimeError):
    """The model returned output that failed schema validation."""


@dataclass
class ContextBlock:
    label: str
    hit: Hit
    body: str


def _body(hit: Hit) -> str:
    text = hit.doc.page_content
    return text.split("\n", 1)[1] if text.startswith("[") and "\n" in text else text


def _tokens(text: str) -> set[str]:
    return {_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP}


class SupportAgent:
    def __init__(self, retriever: Retriever, settings: Settings | None = None, prompt_version: str = prompts.CURRENT_VERSION) -> None:
        self.r, self.s, self.prompt_version = retriever, settings or get_settings(), prompt_version
        self.llm = get_chat_model()
        self.chain = self._build_chain()

    # ------------------------------------------------------------------ chain steps
    def _guard(self, x: dict) -> dict:
        q = x["question"]
        return {"blocked": bool(INJECTION.search(q)), "guess": bool(GUESS.search(q))}

    def _condense(self, x: dict) -> str:
        history, q = x["history_raw"], x["question"]
        if not history or x["guard"]["blocked"]:
            return q
        if self.llm is None:
            return heuristic_condense(q, history)
        chain = (prompts.CONDENSE_PROMPT | self.llm | StrOutputParser()).with_retry(
            retry_if_exception_type=transient_errors(), stop_after_attempt=self.s.llm_max_retries, wait_exponential_jitter=True)
        rewritten = chain.invoke({"history_text": history_text(history), "question": q}).strip().strip('"')
        return rewritten or q

    def _retrieve(self, x: dict) -> RetrievalResult:
        if x["guard"]["blocked"]:
            return RetrievalResult(hits=[], best_dense=0.0, passed=False, strategy="skipped", latency_ms=0.0, reason="guard blocked")
        return self.r.retrieve(x["standalone"], filters=x.get("filters"))

    def _context(self, x: dict) -> dict:
        blocks, used, truncated = [], 0, False
        for i, h in enumerate(x["retrieval"].hits[: self.s.max_context_chunks], 1):
            body = _body(h)
            room = self.s.max_context_chars - used
            if room <= 200:
                truncated = True
                break
            if len(body) > room:
                body, truncated = body[:room], True
            m = h.doc.metadata
            blocks.append(ContextBlock(f"S{i}", h, body))
            used += len(body)
        if truncated:
            log.warning("Context budget reached: chunks truncated (limit %d chars)", self.s.max_context_chars)
        text = "\n\n".join(f"[{b.label}] source={b.hit.doc.metadata['source']} location={b.hit.doc.metadata['location']} "
                           f"version={b.hit.doc.metadata['version']} last_updated={b.hit.doc.metadata['last_updated']}\n{b.body}" for b in blocks)
        return {"text": text, "blocks": blocks, "truncated": truncated}

    def _extractive(self, x: dict) -> dict:
        """OFFLINE BASELINE (no LLM): quote the best-matching lines. Cannot reason, refuse adjacent topics or join facts."""
        q = _tokens(x["standalone"])
        scored = []
        for b in x["context"]["blocks"]:
            for line in b.body.split("\n"):
                clean = re.sub(r"^[#\s|\-*]+", "", line).replace("|", " ").strip()
                if len(clean) > 12 and (ov := len(q & _tokens(clean))):
                    scored.append((ov / (len(q) or 1), b.label, clean))
        top = sorted(scored, key=lambda t: -t[0])[:3]
        if not top:
            ans = LLMAnswer(answer=prompts.REFUSAL_NO_CONTEXT.format(support_email=self.s.support_email), sources=[], confidence=Confidence.none, answered=False)
        else:
            order = {b.label: i for i, b in enumerate(x["context"]["blocks"])}
            top.sort(key=lambda t: order[t[1]])
            prefix = "I can't guess. What the documentation says: " if x["guard"]["guess"] else "From our documentation: "
            ans = LLMAnswer(answer=prefix + " ".join(t[2] for t in top), sources=sorted({t[1] for t in top}),
                            confidence=Confidence.high if max(t[0] for t in top) >= 0.6 else Confidence.partial, answered=True)
        return {"raw": None, "parsed": ans, "parsing_error": None}

    def _generate(self, x: dict) -> dict:
        if self.llm is None:
            return self._extractive(x)
        prompt = prompts.answer_prompt(self.prompt_version, self.s.support_email)
        note = "NOTE: The customer is pressing you to guess or skip sources. The grounding rules still apply." if x["guard"]["guess"] else ""
        structured = self.llm.with_structured_output(LLMAnswer, include_raw=True)
        gen = (prompt | structured).with_retry(retry_if_exception_type=transient_errors(), stop_after_attempt=self.s.llm_max_retries,
                                                wait_exponential_jitter=True)
        return gen.invoke({"history": x["history"], "context": x["context"]["text"], "question": x["question"],
                           "standalone": x["standalone"], "note": note})

    def _refusal(self, x: dict) -> SupportResponse:
        blocked = x["guard"]["blocked"]
        text = (prompts.REFUSAL_GUARD if blocked else prompts.REFUSAL_NO_CONTEXT).format(support_email=self.s.support_email)
        flags = ["prompt_injection_blocked"] if blocked else [f"relevance_gate: {x['retrieval'].reason}"]
        return self._response(x, text, [], Confidence.none, False, True, flags)

    def _finalize(self, x: dict) -> SupportResponse:
        out = x["raw_out"]
        parsed: LLMAnswer | None = out.get("parsed")
        if parsed is None:
            raise OutputValidationError(str(out.get("parsing_error")))
        in_tok = out_tok = 0
        if out.get("raw") is not None and getattr(out["raw"], "usage_metadata", None):
            in_tok, out_tok = out["raw"].usage_metadata.get("input_tokens", 0), out["raw"].usage_metadata.get("output_tokens", 0)
        blocks = {b.label: b for b in x["context"]["blocks"]}
        flags, seen, cites = [], set(), []
        for lab in parsed.sources:
            lab = lab.strip("[] ").upper()
            if lab not in blocks:
                flags.append(f"invalid_citation:{lab}")  # model cited a label that was never supplied -> dropped
                continue
            b = blocks[lab]
            m = b.hit.doc.metadata
            key = (m["source"], m["location"])
            if key in seen:
                continue
            seen.add(key)
            cites.append(Citation(label=lab, source=m["source"], title=m["title"], location=m["location"], version=str(m["version"]),
                                  chunk_id=m["chunk_id"], excerpt=self._excerpt(parsed.answer, b.body), score=round(b.hit.score, 3)))
        conf, answered = parsed.confidence, parsed.answered
        if answered and not cites:
            flags.append("answer_without_valid_citation")
            return self._response(x, prompts.ERROR_INVALID_OUTPUT.format(support_email=self.s.support_email), [], Confidence.none, False, True,
                                  flags, in_tok, out_tok)
        if flags and conf == Confidence.high:
            conf = Confidence.partial
        if answered and x["retrieval"].best_dense < self.s.score_threshold + self.s.high_confidence_margin and conf == Confidence.high:
            conf = Confidence.partial  # retrieval evidence is borderline -> never report 'high'
        if not answered:
            conf, cites = Confidence.none, []
        if x["context"]["truncated"]:
            flags.append("context_truncated")
        return self._response(x, parsed.answer, cites, conf, answered, (not answered) or conf == Confidence.none, flags, in_tok, out_tok)

    @staticmethod
    def _excerpt(answer: str, body: str) -> str:
        a = _tokens(answer)
        lines = [re.sub(r"^[#\s|\-*]+", "", ln).replace("|", " ").strip() for ln in body.split("\n")]
        lines = [ln for ln in lines if len(ln) > 10] or [body.strip()]
        best = max(lines, key=lambda ln: len(a & _tokens(ln)))
        return best[:240]

    def _response(self, x, answer, sources, conf, answered, escalate, flags, in_tok=0, out_tok=0) -> SupportResponse:
        r: RetrievalResult = x["retrieval"]
        return SupportResponse(answer=answer, sources=sources, confidence=conf, answered=answered, escalation_required=escalate,
                               original_query=x["question"], rewritten_query=x["standalone"], flags=flags, best_dense=round(r.best_dense, 3),
                               retrieved=[{"chunk_id": h.doc.metadata["chunk_id"], "source": h.doc.metadata["source"], "location": h.doc.metadata["location"],
                                           "dense": round(h.dense, 3) if h.dense is not None else None, "bm25": round(h.bm25, 2) if h.bm25 else None,
                                           "rerank": round(h.rerank, 3) if h.rerank is not None else None} for h in r.hits],
                               input_tokens=in_tok, output_tokens=out_tok)

    def _build_chain(self) -> Runnable:
        generate = RunnablePassthrough.assign(raw_out=RunnableLambda(self._generate)) | RunnableLambda(self._finalize)
        return (RunnablePassthrough.assign(guard=RunnableLambda(self._guard))
                | RunnablePassthrough.assign(standalone=RunnableLambda(self._condense))
                | RunnablePassthrough.assign(retrieval=RunnableLambda(self._retrieve))
                | RunnablePassthrough.assign(context=RunnableLambda(self._context))
                | RunnableBranch((lambda x: x["guard"]["blocked"] or not x["retrieval"].passed, RunnableLambda(self._refusal)), generate))

    # ------------------------------------------------------------------ public API
    def ask(self, question: str, history: list[tuple[str, str]] | None = None, session_id: str | None = None,
            filters: dict | None = None) -> SupportResponse:
        session_id = session_id or uuid.uuid4().hex[:8]
        question = question.strip()
        history = (history or [])[-2 * self.s.history_window_turns:]
        inputs = {"question": question, "history_raw": history, "history": to_messages(history), "filters": filters}
        start, error = time.perf_counter(), None
        try:
            try:
                resp = self.chain.invoke(inputs)
            except (OutputValidationError, ValidationError) as exc:
                log.warning("Structured output failed validation (%s); retrying once", exc)
                resp = self.chain.invoke(inputs)
        except Exception as exc:  # noqa: BLE001 - never let a traceback reach the user
            resp, error = self._error_response(question, exc), f"{type(exc).__name__}: {exc}"
            log.error("Turn failed: %s", error, exc_info=True)
        resp.latency_ms = round((time.perf_counter() - start) * 1000, 1)
        log_turn({"session_id": session_id, "original_query": resp.original_query or question, "rewritten_query": resp.rewritten_query,
                  "chunks": [(c["chunk_id"], c["dense"], c["rerank"]) for c in resp.retrieved], "best_dense": resp.best_dense,
                  "latency_ms": resp.latency_ms, "input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens,
                  "refused": not resp.answered, "confidence": resp.confidence.value, "escalation": resp.escalation_required, "flags": resp.flags,
                  "citations": [c.source for c in resp.sources], "prompt_version": self.prompt_version, "error": error})
        return resp

    def _error_response(self, question: str, exc: Exception) -> SupportResponse:
        if auth_errors() and isinstance(exc, auth_errors()):
            msg = prompts.ERROR_CONFIG
        elif isinstance(exc, (OutputValidationError, ValidationError)):
            msg = prompts.ERROR_INVALID_OUTPUT
        else:
            msg = prompts.ERROR_UNAVAILABLE
        return SupportResponse(answer=msg.format(support_email=self.s.support_email), confidence=Confidence.none, answered=False,
                               escalation_required=True, original_query=question, rewritten_query=question, flags=["error"], error=type(exc).__name__)
