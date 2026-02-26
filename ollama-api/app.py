import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
import httpx

load_dotenv()

AI_BACKEND_URL = os.getenv("AI_BACKEND_URL", "http://127.0.0.1:11434").rstrip("/")
EMOTION_ENABLED = os.getenv("EMOTION_ENABLED", "false").lower() in ("1", "true", "yes", "on")

app = FastAPI(title="Cathy AI Service Ollama API", version="0.1.0")


@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.get(f"{AI_BACKEND_URL}/api/tags")
            backend_status = "up"
    except Exception:
        backend_status = "down"
    return {
        "ok": True,
        "emotion_enabled": EMOTION_ENABLED,
        "backend_status": backend_status
    }


@app.get("/models")
async def models():
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{AI_BACKEND_URL}/api/tags")
            r.raise_for_status()
            data = r.json()
            return {"models": [m["name"] for m in data.get("models", [])]}
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Backend unavailable: {str(e)}")

@app.get("/models/raw")
async def models_raw():
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(f"{AI_BACKEND_URL}/api/tags")
        r.raise_for_status()
        return r.json()

@app.post("/api/generate")
async def generate(request: Request):
    body = await request.json()
    stream = body.get("stream", True)

    try:
        if not stream:
            timeout = httpx.Timeout(connect=10.0, read=None, write=10.0, pool=10.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(f"{AI_BACKEND_URL}/api/generate", json=body)
                r.raise_for_status()
                return r.json()

        timeout = httpx.Timeout(connect=10.0, read=None, write=10.0, pool=10.0)
        client = httpx.AsyncClient(timeout=timeout)

        async def iter_ndjson():
            try:
                async with client.stream("POST", f"{AI_BACKEND_URL}/api/generate", json=body) as r:
                    r.raise_for_status()
                    async for line in r.aiter_lines():
                        if line:
                            yield (line + "\n").encode("utf-8")
            finally:
                await client.aclose()

        return StreamingResponse(iter_ndjson(), media_type="application/x-ndjson")

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Backend unavailable: {str(e)}")


@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    stream = body.get("stream", True)

    try:
        if not stream:
            timeout = httpx.Timeout(connect=10.0, read=None, write=10.0, pool=10.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(f"{AI_BACKEND_URL}/api/chat", json=body)
                r.raise_for_status()
                return r.json()

        timeout = httpx.Timeout(connect=10.0, read=None, write=10.0, pool=10.0)
        client = httpx.AsyncClient(timeout=timeout)

        async def iter_ndjson():
            try:
                async with client.stream("POST", f"{AI_BACKEND_URL}/api/chat", json=body) as r:
                    r.raise_for_status()
                    async for line in r.aiter_lines():
                        if line:
                            yield (line + "\n").encode("utf-8")
            finally:
                await client.aclose()

        return StreamingResponse(iter_ndjson(), media_type="application/x-ndjson")

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Backend unavailable: {str(e)}")
