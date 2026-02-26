import asyncio
import json
import os
import sqlite3
import time
import uuid
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROMPT_COMPOSER_URL = os.getenv("PROMPT_COMPOSER_URL", "").rstrip("/")
OLLAMA_API_CHAT_URL = os.getenv("OLLAMA_API_CHAT_URL", "").rstrip("/")
OLLAMA_API_CHAT_KEY = os.getenv("OLLAMA_API_CHAT_KEY", "")
STATE_DB = os.getenv("STATE_DB", "/state/orchestrator.sqlite")
GLOBAL_WORKERS = int(os.getenv("GLOBAL_WORKERS", "1"))
MAX_QUEUE_PER_SESSION = int(os.getenv("MAX_QUEUE_PER_SESSION", "3"))
MAX_TOTAL_QUEUED = int(os.getenv("MAX_TOTAL_QUEUED", "50"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "180"))

app = FastAPI(title="cathyAI Orchestrator", version="0.1.0")

CONN: Optional[sqlite3.Connection] = None
queue_wakeup = asyncio.Event()
workers_started = False
session_locks: Dict[str, asyncio.Lock] = {}


def now_ms() -> int:
    return int(time.time() * 1000)


def db_connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(STATE_DB), exist_ok=True)
    conn = sqlite3.connect(STATE_DB, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
          job_id TEXT PRIMARY KEY,
          status TEXT NOT NULL,
          source TEXT NOT NULL,
          session_id TEXT NOT NULL,
          task_type TEXT NOT NULL,
          character_id TEXT NOT NULL,
          person_id TEXT,
          external_user_id TEXT,
          platform TEXT NOT NULL,
          priority INTEGER NOT NULL DEFAULT 100,
          coalesce_key TEXT,
          request_json TEXT NOT NULL,
          pending_messages_json TEXT NOT NULL DEFAULT '[]',
          result_text TEXT,
          error_text TEXT,
          created_at_ms INTEGER NOT NULL,
          started_at_ms INTEGER,
          finished_at_ms INTEGER
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status_prio_ctime ON jobs(status, priority DESC, created_at_ms ASC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_session_status ON jobs(session_id, status)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_coalesce ON jobs(status, coalesce_key, created_at_ms DESC)")
    conn.commit()


class SubmitJobRequest(BaseModel):
    source: str = Field(pattern="^(matrix|webui|chainlit|system)$")
    session_id: str
    task_type: str = "chat"
    character_id: str
    person_id: Optional[str] = None
    external_user_id: Optional[str] = None
    platform: str = Field(pattern="^(matrix|webui)$")
    user_message: Optional[str] = None
    messages: Optional[List[Dict[str, str]]] = None
    coalesce: bool = True
    priority: int = 100
    task_inputs: Optional[Dict[str, Any]] = None


class SubmitJobResponse(BaseModel):
    job_id: str
    status: str
    coalesced_into: Optional[str] = None


def is_coalescible(req: SubmitJobRequest) -> bool:
    return req.coalesce and req.task_type == "chat" and bool(req.user_message and req.user_message.strip())


def make_coalesce_key(req: SubmitJobRequest) -> str:
    return f"{req.task_type}|{req.session_id}|{req.character_id}|{req.platform}"


def count_total_queued(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) AS n FROM jobs WHERE status='queued'").fetchone()
    return int(row["n"])


def count_session_queued(conn: sqlite3.Connection, session_id: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM jobs WHERE status='queued' AND session_id=?",
        (session_id,),
    ).fetchone()
    return int(row["n"])


def try_coalesce_into_existing(conn: sqlite3.Connection, req: SubmitJobRequest) -> Optional[str]:
    key = make_coalesce_key(req)
    row = conn.execute(
        """
        SELECT job_id, pending_messages_json
        FROM jobs
        WHERE status='queued' AND coalesce_key=?
        ORDER BY created_at_ms DESC
        LIMIT 1
        """,
        (key,),
    ).fetchone()

    if not row:
        return None

    pending = json.loads(row["pending_messages_json"] or "[]")
    pending.append(
        {
            "ts_ms": now_ms(),
            "role": "user",
            "content": req.user_message,
            "external_user_id": req.external_user_id,
        }
    )

    conn.execute(
        "UPDATE jobs SET pending_messages_json=? WHERE job_id=?",
        (json.dumps(pending, ensure_ascii=False), row["job_id"]),
    )
    conn.commit()
    return str(row["job_id"])


def insert_job(conn: sqlite3.Connection, req: SubmitJobRequest) -> str:
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    req_payload = req.model_dump()

    pending_messages = []
    if req.user_message and req.user_message.strip():
        pending_messages.append(
            {
                "ts_ms": now_ms(),
                "role": "user",
                "content": req.user_message.strip(),
                "external_user_id": req.external_user_id,
            }
        )

    coalesce_key = make_coalesce_key(req) if is_coalescible(req) else None

    conn.execute(
        """
        INSERT INTO jobs (
          job_id, status, source, session_id, task_type, character_id,
          person_id, external_user_id, platform, priority, coalesce_key,
          request_json, pending_messages_json, created_at_ms
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            job_id,
            "queued",
            req.source,
            req.session_id,
            req.task_type,
            req.character_id,
            req.person_id,
            req.external_user_id,
            req.platform,
            req.priority,
            coalesce_key,
            json.dumps(req_payload, ensure_ascii=False),
            json.dumps(pending_messages, ensure_ascii=False),
            now_ms(),
        ),
    )
    conn.commit()
    return job_id


def pick_next_queued_job(conn: sqlite3.Connection) -> Optional[sqlite3.Row]:
    return conn.execute(
        """
        SELECT *
        FROM jobs
        WHERE status='queued'
        ORDER BY priority DESC, created_at_ms ASC
        LIMIT 1
        """
    ).fetchone()


async def call_prompt_composer(job: sqlite3.Row) -> Dict[str, Any]:
    req = json.loads(job["request_json"])
    pending_messages = json.loads(job["pending_messages_json"] or "[]")

    messages = req.get("messages") or []

    if job["task_type"] == "chat":
        user_parts = [
            m.get("content", "").strip()
            for m in pending_messages
            if m.get("role") == "user" and str(m.get("content", "")).strip()
        ]
        if user_parts:
            combined_user_message = "\n".join(user_parts)
            messages = [{"role": "user", "content": combined_user_message}]

    payload = {
        "character_id": job["character_id"],
        "platform": job["platform"],
        "task": job["task_type"],
        "person_context": {
            "person_id": job["person_id"],
            "external_user_id": job["external_user_id"],
        },
        "memory_context": None,
        "messages": messages,
        "task_inputs": req.get("task_inputs") or {},
    }

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        r = await client.post(f"{PROMPT_COMPOSER_URL}/v1/prompt/compose", json=payload)
        r.raise_for_status()
        return r.json()


async def call_OLLAMA_API(job: sqlite3.Row, prompt_bundle: Dict[str, Any]) -> str:
    if not OLLAMA_API_CHAT_URL:
        raise RuntimeError("OLLAMA_API_CHAT_URL not configured")

    req = json.loads(job["request_json"])
    task_inputs = req.get("task_inputs") or {}

    payload = {
        "model": task_inputs.get("model"),
        "messages": [{"role": "system", "content": prompt_bundle["system_text"]}] + prompt_bundle.get("messages", []),
        "stream": False,
    }

    headers = {"Content-Type": "application/json"}
    if OLLAMA_API_CHAT_KEY:
        headers["Authorization"] = f"Bearer {OLLAMA_API_CHAT_KEY}"

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        r = await client.post(OLLAMA_API_CHAT_URL, json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()

        if isinstance(data, dict):
            if isinstance(data.get("reply"), str):
                return data["reply"]
            if isinstance(data.get("message"), dict):
                return str(data["message"].get("content") or "")
            if isinstance(data.get("response"), str):
                return data["response"]

        return json.dumps(data, ensure_ascii=False)


async def process_one_job(job_id: str):
    global CONN
    assert CONN is not None

    job = CONN.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
    if not job:
        return

    lock = session_locks.setdefault(job["session_id"], asyncio.Lock())
    async with lock:
        job = CONN.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if not job or job["status"] != "queued":
            return

        CONN.execute(
            "UPDATE jobs SET status='running', started_at_ms=? WHERE job_id=?",
            (now_ms(), job_id),
        )
        CONN.commit()

        try:
            prompt_bundle = await call_prompt_composer(job)
            result_text = await call_OLLAMA_API(job, prompt_bundle)

            CONN.execute(
                """
                UPDATE jobs
                SET status='done', result_text=?, finished_at_ms=?
                WHERE job_id=?
                """,
                (result_text, now_ms(), job_id),
            )
            CONN.commit()

        except Exception as e:
            CONN.execute(
                """
                UPDATE jobs
                SET status='error', error_text=?, finished_at_ms=?
                WHERE job_id=?
                """,
                (str(e), now_ms(), job_id),
            )
            CONN.commit()


async def worker_loop(name: str):
    global CONN
    assert CONN is not None

    while True:
        job = pick_next_queued_job(CONN)
        if not job:
            queue_wakeup.clear()
            await queue_wakeup.wait()
            continue

        await process_one_job(str(job["job_id"]))


@app.on_event("startup")
async def on_startup():
    global CONN, workers_started

    CONN = db_connect()
    init_db(CONN)

    if not workers_started:
        for i in range(max(1, GLOBAL_WORKERS)):
            asyncio.create_task(worker_loop(f"worker-{i+1}"))
        workers_started = True


@app.get("/health")
def health():
    return {
        "ok": True,
        "prompt_composer_url": PROMPT_COMPOSER_URL,
        "ollama_api_chat_url": OLLAMA_API_CHAT_URL,
        "global_workers": GLOBAL_WORKERS,
    }


@app.post("/v1/jobs/submit", response_model=SubmitJobResponse)
async def submit_job(req: SubmitJobRequest):
    global CONN
    assert CONN is not None

    if count_total_queued(CONN) >= MAX_TOTAL_QUEUED:
        raise HTTPException(status_code=429, detail="Queue is full")
    if count_session_queued(CONN, req.session_id) >= MAX_QUEUE_PER_SESSION:
        raise HTTPException(status_code=429, detail="Too many queued jobs for this session")

    if is_coalescible(req):
        existing_job_id = try_coalesce_into_existing(CONN, req)
        if existing_job_id:
            queue_wakeup.set()
            return SubmitJobResponse(job_id=existing_job_id, status="queued", coalesced_into=existing_job_id)

    job_id = insert_job(CONN, req)
    queue_wakeup.set()
    return SubmitJobResponse(job_id=job_id, status="queued", coalesced_into=None)


@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str):
    global CONN
    assert CONN is not None

    row = CONN.execute(
        """
        SELECT
          job_id, status, source, session_id, task_type, character_id,
          person_id, external_user_id, platform, priority,
          result_text, error_text,
          created_at_ms, started_at_ms, finished_at_ms
        FROM jobs
        WHERE job_id=?
        """,
        (job_id,),
    ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Job not found")

    return dict(row)
