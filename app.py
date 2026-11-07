"""Streamlit UI: streamlit run app.py   (customer chat + admin console; calls the shared service layer in-process)"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src import services
from src.config import get_settings
from src.db import init_db
from src.retrieval.vector_store import read_manifest
from src.utils.logging_setup import setup_logging

st.set_page_config(page_title="SupportPearlz", page_icon="💧", layout="wide")
setup_logging()
init_db()
services.ensure_admin()
S = get_settings()
st.markdown("""<style>
.block-container{padding-top:1.5rem;max-width:1100px}
.pearl-badge{display:inline-block;padding:2px 10px;border-radius:12px;font-size:.78rem;font-weight:600}
.high{background:#d1fae5;color:#065f46}.partial{background:#fef3c7;color:#92400e}.none{background:#fee2e2;color:#991b1b}
</style>""", unsafe_allow_html=True)
SUGGESTED = ["How long is the AquaPearl 500 Pro warranty?", "My purifier beeps three times and the light flashes red. What should I do?",
             "Can I return it?", "How long does delivery to Skardu take?"]


def guard(fn, *a, **kw):
    """Run a service call and show a friendly error instead of a traceback."""
    try:
        return fn(*a, **kw)
    except services.ServiceError as e:
        st.error(e.message)
    except Exception:  # noqa: BLE001
        st.error("Something went wrong. Please try again.")
    return None


# ------------------------------------------------------------------ auth screens
def auth_screen() -> None:
    st.title("💧 SupportPearlz")
    st.caption("Pearlz Home Systems customer support assistant")
    tab_in, tab_up = st.tabs(["Sign in", "Create account"])
    with tab_in:
        email, pw = st.text_input("Email", key="li_e"), st.text_input("Password", type="password", key="li_p")  # type=password has a built-in show/hide toggle
        if st.button("Sign in", type="primary"):
            res = guard(services.login, email, pw)
            if res:
                st.session_state.update(token=res["token"], user=res["user"], cid=None)
                st.rerun()
    with tab_up:
        name, em = st.text_input("Full name"), st.text_input("Email", key="su_e")
        p1, p2 = st.text_input("Password (8+ chars, letters and digits)", type="password", key="su_p1"), st.text_input("Confirm password", type="password", key="su_p2")
        if st.button("Create account"):
            if guard(services.signup, name, em, p1, p2):
                st.success("Account created. You can sign in now.")


# ------------------------------------------------------------------ customer chat
def show_response(meta: dict, msg_id: int | None, key: str) -> None:
    conf = meta.get("confidence", "none")
    st.markdown(f"<span class='pearl-badge {conf}'>confidence: {conf}</span>" + ("  &nbsp;🧑‍💼 *escalate to human support*" if meta.get("escalation_required") else ""), unsafe_allow_html=True)
    if meta.get("rewritten_query") and meta["rewritten_query"] != meta.get("original_query"):
        st.caption(f"Understood as: {meta['rewritten_query']}")
    srcs = meta.get("sources", [])
    if srcs:
        with st.expander(f"Sources ({len(srcs)})"):
            for c in srcs:
                st.markdown(f"**{c['source']}** — {c['location']} · v{c['version']}  \n> {c['excerpt']}  \n`retrieval score {c.get('score')}`")
    else:
        st.caption("No supporting source found.")
    if msg_id:
        a, b, _ = st.columns([1, 1, 8])
        if a.button("👍 Helpful", key=f"up{key}"):
            guard(services.set_feedback, st.session_state.user, msg_id, 1)
            st.toast("Thanks for the feedback")
        if b.button("👎 Not helpful", key=f"dn{key}"):
            guard(services.set_feedback, st.session_state.user, msg_id, -1)
            st.toast("Thanks, we'll review this answer")


def chat_page() -> None:
    user = st.session_state.user
    convs = services.list_conversations(user)
    with st.sidebar:
        if st.button("➕ New chat", width="stretch"):
            st.session_state.cid = None
            st.rerun()
        st.caption("Your conversations")
        for c in convs:
            col1, col2 = st.columns([5, 1])
            if col1.button(c["title"][:32] or "Chat", key=f"c{c['id']}", width="stretch"):
                st.session_state.cid = c["id"]
                st.rerun()
            if col2.button("🗑", key=f"d{c['id']}"):
                guard(services.delete_conversation, user, c["id"])
                st.session_state.cid = None
                st.rerun()
        with st.expander("Filters"):
            pl = st.selectbox("Product", ["Any", "AquaPearl 500 Pro", "AquaPearl 300", "AeroPearl Mini"])
            dt = st.selectbox("Document type", ["Any", "policy", "manual", "guide", "faq", "pricing"])
    st.title("Ask SupportPearlz")
    cid = st.session_state.get("cid")
    if cid:
        for m in guard(services.get_conversation, user, cid) or []:
            with st.chat_message(m["role"]):
                st.write(m["content"])
                if m["role"] == "assistant" and m["meta"]:
                    show_response(m["meta"], m["id"], f"h{m['id']}")
    else:
        st.info("Ask about warranty, returns, delivery, installation, filters or troubleshooting. Answers come only from Pearlz documents and carry sources.")
        cols = st.columns(len(SUGGESTED))
        for col, q in zip(cols, SUGGESTED):
            if col.button(q, width="stretch"):
                st.session_state.pending = q
    q = st.chat_input("Type your question") or st.session_state.pop("pending", None)
    if q:
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"), st.spinner("Searching the documentation..."):
            filters = {k: v for k, v in {"product_line": pl if pl != "Any" else None, "doc_type": dt if dt != "Any" else None}.items() if v}
            out = guard(services.chat, user, q, cid, filters or None)
        if out:
            st.session_state.cid = out["conversation_id"]
            st.rerun()


# ------------------------------------------------------------------ admin
def admin_page() -> None:
    user = st.session_state.user
    st.title("Admin console")
    t = st.tabs(["Overview", "Users", "Knowledge base", "Evaluation", "Logs"])
    with t[0]:
        s = guard(services.admin_stats, user)
        if s:
            c = st.columns(5)
            c[0].metric("Users", s["users"]); c[1].metric("Conversations", s["conversations"]); c[2].metric("Questions", s["questions"])
            c[3].metric("Refused", s["refused"]); c[4].metric("Avg latency (ms)", s["avg_latency_ms"])
            d = st.columns(3)
            d[0].metric("Escalated", s["escalated"]); d[1].metric("👍", s["helpful"]); d[2].metric("👎", s["not_helpful"])
            if s["daily"]:
                st.plotly_chart(px.bar(pd.DataFrame(s["daily"]), x="day", y="questions", title="Questions per day"), width="stretch")
            st.subheader("Recent unanswered questions (knowledge-base gaps)")
            st.write(services.unanswered_questions(user) or "None yet.")
    with t[1]:
        users = services.list_users(user)
        st.dataframe(pd.DataFrame(users), width="stretch")
        uid = st.selectbox("Change role for user id", [u["id"] for u in users])
        role = st.selectbox("New role", ["customer", "admin"])
        if st.button("Update role"):
            if guard(services.set_role, user, uid, role) is None:
                st.success("Role updated")
    with t[2]:
        m = read_manifest() or {}
        st.caption(f"Index: {m.get('chunks')} chunks · embeddings {m.get('embedding', {}).get('tag')} ({m.get('embedding', {}).get('dimensions')}d) · "
                   f"chunking {m.get('chunk_strategy')}/{m.get('chunk_size')}/{m.get('chunk_overlap')} · updated {m.get('updated_at')}")
        st.dataframe(pd.DataFrame(services.list_documents(user)), width="stretch")
        up = st.file_uploader("Upload a document (pdf, docx, md, txt, csv)")
        if up and st.button("Upload and index"):
            res = guard(services.upload_document, user, up.name, up.getvalue())
            if res:
                st.success(f"Indexed. added={res['added']} updated={res['updated']}")
        name = st.text_input("Delete document (file name)")
        if st.button("Delete and re-sync") and name:
            res = guard(services.delete_document, user, name)
            if res:
                st.success(f"Removed: {res['removed']}")
        if st.button("Re-sync index now"):
            st.json(guard(services.reindex, user) or {})
    with t[3]:
        p = Path("evaluation/results/evaluation_results.json")
        if not p.exists():
            st.info("Run `python -m evaluation.run_eval` first.")
        else:
            data = json.loads(p.read_text())
            st.caption(f"Run {data['meta']['run_at']} · LLM {data['meta']['llm_provider']} · prompt {data['meta']['prompt_version']}")
            if data["meta"]["llm_provider"] == "extractive":
                st.warning("Baseline run without an LLM: measures retrieval, gate and guards, not answer quality.")
            st.json(data["aggregate"])
            df = pd.DataFrame(data["cases"])
            st.plotly_chart(px.histogram(df, x="category", color="verdict", barmode="group", title="Verdicts by category"), width="stretch")
            st.dataframe(df[["id", "category", "expected", "answered", "confidence", "verdict", "failure_stage", "latency_ms"]], width="stretch")
    with t[4]:
        lp = S.log_dir / "turns.jsonl"
        if lp.exists():
            rows = [json.loads(line) for line in lp.read_text(encoding="utf-8").splitlines()[-200:]]
            st.dataframe(pd.DataFrame(rows)[["ts", "session_id", "original_query", "rewritten_query", "best_dense", "latency_ms", "refused", "confidence", "flags", "error"]].iloc[::-1],
                         width="stretch")
        else:
            st.info("No turns logged yet.")


# ------------------------------------------------------------------ router
if "token" not in st.session_state:
    auth_screen()
else:
    try:
        st.session_state.user = services.user_from_token(st.session_state.token)
    except services.ServiceError:
        st.session_state.clear()
        st.rerun()
    user = st.session_state.user
    with st.sidebar:
        st.markdown(f"**{user['name']}** · {user['role']}")
        if st.button("Sign out"):
            st.session_state.clear()
            st.rerun()
    page = "Chat"
    if user["role"] == "admin":
        page = st.sidebar.radio("Area", ["Chat", "Admin"])
    chat_page() if page == "Chat" else admin_page()
