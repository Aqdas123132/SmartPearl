"""Service layer shared by the FastAPI API and the Streamlit UI. All authorisation lives here, not in the UI."""
from __future__ import annotations

import json
import logging
import re
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import bcrypt
import jwt
from sqlalchemy import func, select

from src.chains.rag_chain import SupportAgent
from src.chains.schemas import SupportResponse
from src.config import get_settings
from src.db import Conversation, Message, User, session
from src.ingestion.loaders import SUPPORTED
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import delete_source, load_or_build, read_manifest, sync_index

log = logging.getLogger(__name__)


class ServiceError(Exception):
    """A user-presentable error with an HTTP-like status code."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.message, self.status = message, status


# ------------------------------------------------------------------------------ auth
EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _validate_password(pw: str) -> None:
    if len(pw) < 8 or not re.search(r"[A-Za-z]", pw) or not re.search(r"\d", pw):
        raise ServiceError("Password must be at least 8 characters and contain letters and digits.")


def signup(name: str, email: str, password: str, confirm: str, role: str = "customer") -> dict:
    name, email = name.strip(), email.strip().lower()
    if not name or not EMAIL.match(email):
        raise ServiceError("Please enter your full name and a valid email address.")
    if password != confirm:
        raise ServiceError("Passwords do not match.")
    _validate_password(password)
    with session() as db:
        if db.scalar(select(User).where(User.email == email)):
            raise ServiceError("An account with this email already exists.", 409)
        u = User(name=name, email=email, role=role, password_hash=bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode())
        db.add(u)
        db.flush()
        return {"id": u.id, "name": u.name, "email": u.email, "role": u.role}


def login(email: str, password: str) -> dict:
    with session() as db:
        u = db.scalar(select(User).where(User.email == email.strip().lower()))
        if not u or not bcrypt.checkpw(password.encode(), u.password_hash.encode()):
            raise ServiceError("Incorrect email or password.", 401)
        u.last_login = datetime.now(timezone.utc)
        s = get_settings()
        exp = datetime.now(timezone.utc) + timedelta(minutes=s.token_ttl_minutes)
        token = jwt.encode({"sub": str(u.id), "role": u.role, "exp": exp}, s.secret_key, algorithm="HS256")
        return {"token": token, "user": {"id": u.id, "name": u.name, "email": u.email, "role": u.role}}


def user_from_token(token: str) -> dict:
    try:
        data = jwt.decode(token, get_settings().secret_key, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise ServiceError("Your session has expired. Please sign in again.", 401)
    with session() as db:
        u = db.get(User, int(data["sub"]))
        if not u:
            raise ServiceError("Account no longer exists.", 401)
        return {"id": u.id, "name": u.name, "email": u.email, "role": u.role}  # role re-read from DB, not trusted from the token


def require_admin(user: dict) -> None:
    if user["role"] != "admin":
        raise ServiceError("Administrator access required.", 403)


def ensure_admin() -> None:
    s = get_settings()
    if s.admin_email and s.admin_password:
        with session() as db:
            if db.scalar(select(User).where(User.email == s.admin_email.lower())):
                return
        signup("Administrator", s.admin_email, s.admin_password, s.admin_password, role="admin")


# ------------------------------------------------------------------------------ agent lifecycle
_lock = threading.Lock()
_agent: SupportAgent | None = None


def get_agent() -> SupportAgent:
    global _agent
    with _lock:
        if _agent is None:
            _agent = SupportAgent(Retriever(load_or_build()))
        return _agent


def reset_agent() -> None:
    global _agent
    with _lock:
        _agent = None


# ------------------------------------------------------------------------------ chat
def chat(user: dict, question: str, conversation_id: str | None = None, filters: dict | None = None) -> dict:
    question = question.strip()
    if not question:
        raise ServiceError("Please type a question.")
    if len(question) > 1000:
        raise ServiceError("Questions are limited to 1000 characters.")
    with session() as db:
        conv = db.get(Conversation, conversation_id) if conversation_id else None
        if conv and conv.user_id != user["id"]:
            raise ServiceError("Conversation not found.", 404)  # session isolation: never reveal other users' ids
        if conv is None:
            conv = Conversation(id=str(uuid.uuid4()), user_id=user["id"], title=question[:60])
            db.add(conv)
            db.flush()
        history = [(m.role, m.content) for m in conv.messages]
        cid = conv.id
    resp: SupportResponse = get_agent().ask(question, history, session_id=cid[:8], filters=filters)
    with session() as db:
        db.add(Message(conversation_id=cid, role="user", content=question))
        m = Message(conversation_id=cid, role="assistant", content=resp.answer, answered=resp.answered, escalated=resp.escalation_required,
                    latency_ms=resp.latency_ms, meta=json.loads(resp.model_dump_json(exclude={"answer"})))
        db.add(m)
        db.flush()
        return {"conversation_id": cid, "message_id": m.id, "response": resp}


def list_conversations(user: dict) -> list[dict]:
    with session() as db:
        rows = db.scalars(select(Conversation).where(Conversation.user_id == user["id"]).order_by(Conversation.created_at.desc())).all()
        return [{"id": c.id, "title": c.title, "created_at": c.created_at, "messages": len(c.messages)} for c in rows]


def get_conversation(user: dict, cid: str) -> list[dict]:
    with session() as db:
        c = db.get(Conversation, cid)
        if not c or (c.user_id != user["id"] and user["role"] != "admin"):
            raise ServiceError("Conversation not found.", 404)
        return [{"id": m.id, "role": m.role, "content": m.content, "meta": m.meta, "feedback": m.feedback, "created_at": m.created_at} for m in c.messages]


def delete_conversation(user: dict, cid: str) -> None:
    with session() as db:
        c = db.get(Conversation, cid)
        if not c or c.user_id != user["id"]:
            raise ServiceError("Conversation not found.", 404)
        db.delete(c)


def set_feedback(user: dict, message_id: int, value: int) -> None:
    with session() as db:
        m = db.get(Message, message_id)
        if not m or m.conversation.user_id != user["id"]:
            raise ServiceError("Message not found.", 404)
        m.feedback = 1 if value > 0 else -1


# ------------------------------------------------------------------------------ admin
def admin_stats(user: dict) -> dict:
    require_admin(user)
    with session() as db:
        asst = Message.role == "assistant"
        n_ans = db.scalar(select(func.count()).where(asst)) or 0
        refused = db.scalar(select(func.count()).where(asst, Message.answered.is_(False))) or 0
        return {
            "users": db.scalar(select(func.count(User.id))) or 0, "conversations": db.scalar(select(func.count(Conversation.id))) or 0,
            "questions": db.scalar(select(func.count()).where(Message.role == "user")) or 0, "assistant_messages": n_ans, "refused": refused,
            "escalated": db.scalar(select(func.count()).where(asst, Message.escalated.is_(True))) or 0,
            "helpful": db.scalar(select(func.count()).where(Message.feedback == 1)) or 0,
            "not_helpful": db.scalar(select(func.count()).where(Message.feedback == -1)) or 0,
            "avg_latency_ms": round(db.scalar(select(func.avg(Message.latency_ms)).where(asst)) or 0, 1),
            "daily": [{"day": str(d), "questions": n} for d, n in db.execute(
                select(func.date(Message.created_at), func.count()).where(Message.role == "user").group_by(func.date(Message.created_at))).all()],
        }


def unanswered_questions(user: dict, limit: int = 50) -> list[str]:
    require_admin(user)
    with session() as db:
        out = []
        for m in db.scalars(select(Message).where(Message.role == "assistant", Message.answered.is_(False)).order_by(Message.id.desc()).limit(limit)):
            prev = db.scalar(select(Message).where(Message.conversation_id == m.conversation_id, Message.id < m.id, Message.role == "user").order_by(Message.id.desc()))
            if prev:
                out.append(prev.content)
        return out


def list_users(user: dict) -> list[dict]:
    require_admin(user)
    with session() as db:
        return [{"id": u.id, "name": u.name, "email": u.email, "role": u.role, "created_at": u.created_at, "last_login": u.last_login,
                 "conversations": len(u.conversations)} for u in db.scalars(select(User).order_by(User.id))]


def set_role(user: dict, user_id: int, role: str) -> None:
    require_admin(user)
    if role not in ("admin", "customer"):
        raise ServiceError("Role must be 'admin' or 'customer'.")
    with session() as db:
        u = db.get(User, user_id)
        if not u:
            raise ServiceError("User not found.", 404)
        if u.id == user["id"] and role != "admin":
            raise ServiceError("You cannot remove your own admin role.")
        u.role = role


def list_documents(user: dict) -> list[dict]:
    require_admin(user)
    kb = get_settings().knowledge_base_dir
    manifest = read_manifest() or {"sources": {}}
    return [{"file": p.relative_to(kb).as_posix(), "size_kb": round(p.stat().st_size / 1024, 1),
             "indexed_chunks": manifest["sources"].get(p.relative_to(kb).as_posix(), {}).get("chunks", 0),
             "version": manifest["sources"].get(p.relative_to(kb).as_posix(), {}).get("version", "-")}
            for p in sorted(kb.rglob("*")) if p.is_file() and p.name != "manifest.json"]


def upload_document(user: dict, filename: str, data: bytes) -> dict:
    require_admin(user)
    s = get_settings()
    name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(filename).name)
    if Path(name).suffix.lower() not in SUPPORTED:
        raise ServiceError(f"Unsupported file type. Allowed: {', '.join(sorted(SUPPORTED))}.")
    if len(data) > s.max_upload_mb * 1024 * 1024:
        raise ServiceError(f"File is larger than {s.max_upload_mb} MB.", 413)
    (s.knowledge_base_dir / name).write_bytes(data)
    return reindex(user)


def delete_document(user: dict, filename: str) -> dict:
    require_admin(user)
    kb = get_settings().knowledge_base_dir
    path = (kb / Path(filename).name)
    if not path.exists():
        raise ServiceError("Document not found.", 404)
    path.unlink()
    return reindex(user)


def reindex(user: dict) -> dict:
    """Incremental sync (delete-by-source and reinsert changed files), then refresh the BM25 view of the live retriever."""
    require_admin(user)
    agent = get_agent()
    result = sync_index(agent.r.store)
    agent.r.refresh()
    return {k: result[k] for k in ("added", "updated", "removed")} | {"ingestion": result["report"]["summary"]}
