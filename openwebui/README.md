# OpenWebUI Setup for HistoriCon

## Prerequisites
- Docker Desktop installed and running
- `uv` installed (`brew install uv` or `curl -Lsf https://astral.sh/uv | sh`)
- ChromaDB populated (`uv run python scripts/create_embeddings.py`)

---

## 1. Start the MCP server

```bash
# From the project root
secrets && se          # load environment secrets
uv run python agents/mcp_server.py
# → Listening on http://0.0.0.0:8001/mcp
```

Keep this terminal open. The server must be running before OpenWebUI can use it.

---

## 2. Start OpenWebUI

```bash
docker compose up -d
# → http://localhost:3000
```

On first launch, OpenWebUI will prompt you to create an **admin account**. This account can invite other users.

---

## 3. Connect the MCP server

1. Log in as admin → **Settings** (top-right gear icon) → **Admin Panel**
2. Go to **Connections** → **MCP Servers**
3. Click **+ Add MCP Server**
4. Set the URL to: `http://host.docker.internal:8001/mcp`
5. Click **Save** — you should see the 4 HistoriCon tools listed

> `host.docker.internal` is a special Docker DNS name that resolves to your Mac's host IP, allowing the container to reach the MCP server running locally.

---

## 4. Apply the system prompt

1. In Admin Panel → **Models** → select the model you want to use (e.g. Claude, GPT-4o, Llama)
2. Scroll to **System Prompt**
3. Paste the contents of [system_prompt.md](system_prompt.md)
4. Save

Alternatively, users can apply the system prompt per-conversation via the chat settings icon.

---

## 5. Invite users

1. Admin Panel → **Users** → **+ Invite User**
2. Enter their email address; they receive a signup link
3. New users sign up with their own password
4. Only invited users can create accounts (enforced by `WEBUI_AUTH=true`)

---

## 6. Using the chat

1. Open a chat at `http://localhost:3000`
2. Select a model that supports tool calling (e.g. Claude Sonnet, GPT-4o, Llama 3.3)
3. Ask questions in Greek or English:
   - *"Πες μου για το επεισόδιο του Κοσκωτά"*
   - *"What episodes cover the 1821 Greek Revolution?"*
   - *"Who are the hosts of HistoriCon?"*

The model will automatically call `search_documents` and cite episode names + timestamps.

---

## Stopping services

```bash
docker compose down   # stop OpenWebUI (data persists in the open-webui volume)
# Ctrl+C in the MCP server terminal to stop it
```

---

## Deployment note

This setup runs locally on `localhost`. To expose it to remote users, see the planned Phase 2 deployment notes (Tailscale or Caddy reverse proxy). Do not expose port 3000 or 8001 directly to the internet without adding TLS and additional authentication.
