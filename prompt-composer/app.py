import os
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

CHAR_API_URL = os.getenv("CHAR_API_URL", "").rstrip("/")
CHAR_API_KEY = os.getenv("CHAR_API_KEY", "")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "10"))

app = FastAPI(title="cathyAI Prompt Composer", version="0.1.0")


class MessageItem(BaseModel):
    role: str
    content: str


class ComposeRequest(BaseModel):
    character_id: str
    platform: str = Field(pattern="^(webui|matrix)$")
    task: str
    person_context: Optional[Dict[str, Any]] = None
    memory_context: Optional[Dict[str, Any]] = None
    messages: List[MessageItem] = []
    task_inputs: Optional[Dict[str, Any]] = None


class ComposeResponse(BaseModel):
    character_id: str
    platform: str
    task: str
    system_sections: List[Dict[str, str]]
    system_text: str
    messages: List[MessageItem]


async def fetch_character_private(character_id: str) -> Dict[str, Any]:
    if not CHAR_API_URL:
        raise HTTPException(status_code=500, detail="CHAR_API_URL not configured")

    headers = {}
    if CHAR_API_KEY:
        headers["x-api-key"] = CHAR_API_KEY

    url = f"{CHAR_API_URL}/characters/{character_id}?view=private"

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        r = await client.get(url, headers=headers)
        if r.status_code == 404:
            raise HTTPException(status_code=404, detail=f"Character not found: {character_id}")
        if r.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"Character API error {r.status_code}: {r.text[:200]}")
        return r.json()


def build_task_rules(task: str, platform: str, task_inputs: Optional[Dict[str, Any]]) -> str:
    task_inputs = task_inputs or {}

    if task == "chat":
        return (
            "You are responding in an interactive conversation.\n"
            "Stay in character while remaining helpful and coherent.\n"
            "If you are unsure, say so clearly.\n"
            "Do not fabricate facts."
        )

    if task == "news_digest":
        return (
            "You are formatting a news summary for a Matrix room.\n"
            "Use only the provided facts/headlines/context.\n"
            "Do not invent details.\n"
            "Prefer concise bullets or short sections."
        )

    if task == "cleanup_summary":
        return (
            "You are formatting a storage/media cleanup summary.\n"
            "Use only the provided metrics and reasons.\n"
            "Do not invent counts, sizes, or actions.\n"
            "Keep the message short and clear."
        )

    if task == "remove_ack":
        return (
            "You are acknowledging a message removal/moderation action.\n"
            "Confirm clearly and briefly.\n"
            "Do not expose internal implementation details."
        )

    return "Follow the provided task context and respond accurately."


def build_identity_section(person_context: Optional[Dict[str, Any]]) -> str:
    if not person_context:
        return ""

    lines = ["Current speaker context:"]
    for key in ("person_id", "preferred_name", "external_user_id"):
        val = person_context.get(key)
        if val:
            lines.append(f"- {key}: {val}")
    return "\n".join(lines)


def build_memory_section(memory_context: Optional[Dict[str, Any]]) -> str:
    if not memory_context:
        return ""

    facts = memory_context.get("facts") or []
    snippets = memory_context.get("snippets") or []

    parts: List[str] = []
    if facts:
        parts.append("[Known facts]")
        for item in facts[:20]:
            parts.append(f"- {str(item)}")

    if snippets:
        parts.append("\n[Relevant past context]")
        for item in snippets[:10]:
            parts.append(f"- {str(item)}")

    return "\n".join(parts).strip()


def extract_prompt_fragments(char: Dict[str, Any], platform: str) -> Dict[str, str]:
    prompts = char.get("prompts") or {}

    persona = prompts.get("system") or char.get("system_prompt") or ""
    background = prompts.get("background") or char.get("character_background") or ""

    matrix_rules = (
        prompts.get("matrix_append_rules")
        or ((char.get("matrix") or {}).get("append_rules_text"))
        or ""
    )
    webui_rules = (
        prompts.get("webui_append_rules")
        or ((char.get("webui") or {}).get("append_rules_text"))
        or ""
    )

    platform_rules = matrix_rules if platform == "matrix" else webui_rules

    return {
        "persona": str(persona).strip(),
        "background": str(background).strip(),
        "platform_rules": str(platform_rules).strip(),
    }


@app.get("/health")
def health():
    return {"ok": True, "char_api_url": CHAR_API_URL}


@app.post("/v1/prompt/compose", response_model=ComposeResponse)
async def compose(req: ComposeRequest):
    char = await fetch_character_private(req.character_id)
    frags = extract_prompt_fragments(char, req.platform)

    sections: List[Dict[str, str]] = []

    def add(name: str, text: str):
        text = (text or "").strip()
        if text:
            sections.append({"name": name, "text": text})

    add("persona", frags["persona"])
    add("background", frags["background"])
    add("platform_rules", frags["platform_rules"])
    add("task_rules", build_task_rules(req.task, req.platform, req.task_inputs))
    add("identity", build_identity_section(req.person_context))
    add("memory", build_memory_section(req.memory_context))

    system_text = "\n\n".join(f"[{s['name']}]\n{s['text']}" for s in sections)

    return ComposeResponse(
        character_id=req.character_id,
        platform=req.platform,
        task=req.task,
        system_sections=sections,
        system_text=system_text,
        messages=req.messages,
    )
