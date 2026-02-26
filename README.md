# cathyAI-infra

AI-side orchestration services for cathyAI, running on the ASUS AI node.

## Services

- **ollama-api** (port 8100): FastAPI proxy for AI backends (Ollama, NPU-SVC)
- **prompt-composer** (port 8110): Builds unified prompts from character + platform + task + memory/identity context
- **ai-orchestrator** (port 8120): Queue + coalescing + worker scheduler for AI jobs

## External Dependencies

- Character API (192.168.1.59:8090)
- Identity API  (192.168.1.59:8092)
- Ollama        (127.0.0.1:11434)

## Setup

```bash
# Copy and configure environment files
cp ollama-api/.env.example ollama-api/.env
cp prompt-composer/.env.example prompt-composer/.env
cp ai-orchestrator/.env.example ai-orchestrator/.env

# Edit .env files with actual API keys and endpoints

# Bring up services
docker-compose up -d --build
```

## Health Checks

```bash
curl http://127.0.0.1:8100/health
curl http://127.0.0.1:8110/health
curl http://127.0.0.1:8120/health
```

## API Usage

### Ollama API

```bash
# List models
curl http://127.0.0.1:8100/models

# Generate text (streaming)
curl -N http://127.0.0.1:8100/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model":"llama2","prompt":"Hello"}'

# Chat (non-streaming)
curl http://127.0.0.1:8100/api/chat \
  -H "Content-Type: application/json" \
  -d '{"model":"llama2","messages":[{"role":"user","content":"Hi"}],"stream":false}'
```

### Prompt Composer

```bash
curl -X POST http://127.0.0.1:8110/v1/prompt/compose \
  -H "Content-Type: application/json" \
  -d '{
    "character_id": "catherine",
    "platform": "webui",
    "task": "chat",
    "person_context": {"person_id": "p_user", "preferred_name": "User"},
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### AI Orchestrator

```bash
# Submit job
curl -X POST http://127.0.0.1:8120/v1/jobs/submit \
  -H "Content-Type: application/json" \
  -d '{
    "source": "webui",
    "session_id": "test-session-1",
    "task_type": "chat",
    "character_id": "catherine",
    "platform": "webui",
    "user_message": "Hello, who are you?",
    "coalesce": true
  }'

# Check job status
curl http://127.0.0.1:8120/v1/jobs/{job_id}
```

## Features

### Message Coalescing

Multiple rapid messages to the same session are automatically coalesced into a single AI request, reducing redundant processing.

### Queue Management

- Per-session queue limits (default: 3)
- Global queue limits (default: 50)
- Priority-based processing
- Single worker by default (configurable)
