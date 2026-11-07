"""FastAPI service. Run: uvicorn src.api:app --port 8000"""
from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src import services
from src.config import get_settings
from src.db import init_db
from src.retrieval.vector_store import read_manifest
from src.utils.logging_setup import setup_logging

setup_logging()
init_db()
services.ensure_admin()
app = FastAPI(title="SupportPearlz API", version="1.0")
_hits: dict[int, deque] = defaultdict(deque)


@app.exception_handler(services.ServiceError)
async def _service_error(_, exc: services.ServiceError):
    return JSONResponse(status_code=exc.status, content={"detail": exc.message})


@app.exception_handler(Exception)
async def _unhandled(_, exc: Exception):  # never expose a traceback
    return JSONResponse(status_code=500, content={"detail": "Something went wrong. Please try again."})


def current_user(authorization: str = Header(default="")) -> dict:
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Missing bearer token.")
    return services.user_from_token(authorization.split(" ", 1)[1])


def admin_user(user: dict = Depends(current_user)) -> dict:
    services.require_admin(user)
    return user


def rate_limit(user: dict = Depends(current_user)) -> dict:
    q, now, limit = _hits[user["id"]], time.time(), get_settings().rate_limit_per_minute
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= limit:
        raise HTTPException(429, "Too many requests. Please wait a moment.")
    q.append(now)
    return user


class SignupIn(BaseModel):
    name: str
    email: str
    password: str
    confirm_password: str


class LoginIn(BaseModel):
    email: str
    password: str


class ChatIn(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    conversation_id: str | None = None
    product_line: str | None = None
    doc_type: str | None = None


class FeedbackIn(BaseModel):
    message_id: int
    helpful: bool


@app.get("/health")
def health() -> dict:
    m = read_manifest()
    return {"status": "ok" if m else "index_missing", "index_chunks": (m or {}).get("chunks", 0), "llm_provider": get_settings().llm_provider,
            "embedding": (m or {}).get("embedding", {}).get("tag")}


@app.post("/auth/signup", status_code=201)
def signup(body: SignupIn) -> dict:
    return services.signup(body.name, body.email, body.password, body.confirm_password)


@app.post("/auth/login")
def login(body: LoginIn) -> dict:
    return services.login(body.email, body.password)


@app.post("/chat")
async def chat(body: ChatIn, user: dict = Depends(rate_limit)) -> dict:
    filters = {k: v for k, v in {"product_line": body.product_line, "doc_type": body.doc_type}.items() if v}
    try:
        out = await asyncio.wait_for(run_in_threadpool(services.chat, user, body.question, body.conversation_id, filters or None),
                                     timeout=get_settings().request_timeout_s)
    except asyncio.TimeoutError:
        raise HTTPException(504, "The request took too long. Please try again.")
    return {"conversation_id": out["conversation_id"], "message_id": out["message_id"], **out["response"].model_dump(mode="json")}


@app.get("/conversations")
def conversations(user: dict = Depends(current_user)) -> list[dict]:
    return services.list_conversations(user)


@app.get("/conversations/{cid}")
def conversation(cid: str, user: dict = Depends(current_user)) -> list[dict]:
    return services.get_conversation(user, cid)


@app.delete("/conversations/{cid}", status_code=204)
def delete_conversation(cid: str, user: dict = Depends(current_user)) -> None:
    services.delete_conversation(user, cid)


@app.post("/feedback")
def feedback(body: FeedbackIn, user: dict = Depends(current_user)) -> dict:
    services.set_feedback(user, body.message_id, 1 if body.helpful else -1)
    return {"ok": True}


@app.get("/admin/stats")
def stats(user: dict = Depends(admin_user)) -> dict:
    return services.admin_stats(user)


@app.get("/admin/users")
def users(user: dict = Depends(admin_user)) -> list[dict]:
    return services.list_users(user)


@app.get("/admin/documents")
def documents(user: dict = Depends(admin_user)) -> list[dict]:
    return services.list_documents(user)


@app.post("/admin/documents")
async def upload(file: UploadFile = File(...), user: dict = Depends(admin_user)) -> dict:
    return await run_in_threadpool(services.upload_document, user, file.filename or "", await file.read())


@app.delete("/admin/documents/{name}")
def remove(name: str, user: dict = Depends(admin_user)) -> dict:
    return services.delete_document(user, name)


@app.post("/admin/reindex")
def reindex(user: dict = Depends(admin_user)) -> dict:
    return services.reindex(user)
