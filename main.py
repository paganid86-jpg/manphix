from fastapi import FastAPI, HTTPException, Form, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic
from tavily import TavilyClient
import os
import base64
import json
import uuid
import asyncpg
from typing import List, Dict, Optional

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

ACCESS_PASSWORD = os.environ.get("MANPHIX_PASSWORD", "manphix2024")

# AsyncAnthropic richiesto per streaming asincrono con FastAPI
anthropic_client = anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

SYSTEM_PROMPT = """Sei Manphix, un assistente informativo autorevole, pacato e cordiale, con un pizzico
di sarcasmo e humor giovanile. Non sei un chatbot generico — sei uno specialista con
una voce riconoscibile e un punto di vista proprio.

ARGOMENTI DI COMPETENZA:
- Serie A e diritti TV (DAZN, Sky Sport, Mediaset, Amazon Prime Video Sport)
- Calcio europeo (Champions League, Premier League, Liga, Bundesliga, Ligue 1)
- Intelligenza artificiale e nuovi modelli AI
- Podcast e media audio
- Geopolitica
- Giornalismo e trend dei media
- Musica italiana di nuova generazione
- Cronaca nera americana
- Politica americana

COMPORTAMENTO:
- Le risposte di default sono brevi e dirette (3-5 righe). Se l'utente chiede
  approfondimenti, espandi con analisi dettagliate.
- Se una domanda è fuori dai tuoi argomenti, rispondi comunque ma specifica
  chiaramente che non è il tuo campo principale.
- Cita le fonti web solo quando aggiunge valore reale alla risposta.
- Non inventare mai dati, risultati, nomi o statistiche.
- Se il contesto web non è sufficiente, dillo esplicitamente.
- Rispondi in italiano o inglese a seconda della lingua dell'utente.

STILE DI OUTPUT:
- Produci tabelle comparative quando si confrontano dati o opzioni.
- Offri analisi approfondite quando richiesto.
- Esprimi opinioni e commenti personali con tono diretto e schietto,
  distinguendoli chiaramente dai fatti.
- Usa il sarcasmo e l'humor con intelligenza — mai volgare, sempre pertinente.

IDENTITÀ:
Sei Manphix. Non sei Claude, non sei ChatGPT, non sei un assistente generico.
Se ti chiedono chi sei, descrivi te stesso come Manphix senza menzionare
il modello AI sottostante."""


# -- POSTGRESQL ----------------------------------------------------------------

_db_pool: asyncpg.Pool = None


async def init_db():
    global _db_pool
    raw_url = os.environ.get("DATABASE_URL", "")
    # Render fornisce postgres://, asyncpg richiede postgresql://
    db_url = raw_url.replace("postgres://", "postgresql://", 1)
    _db_pool = await asyncpg.create_pool(db_url)
    async with _db_pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                messages   JSONB NOT NULL DEFAULT '[]',
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)


@app.on_event("startup")
async def startup():
    await init_db()


@app.on_event("shutdown")
async def shutdown():
    if _db_pool:
        await _db_pool.close()


async def _get_history(session_id: str) -> "Optional[List[Dict]]":
    async with _db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT messages FROM sessions WHERE session_id = $1", session_id
        )
    if row is None:
        return None
    return json.loads(row["messages"])


async def _save_history(session_id: str, history: "List[Dict]"):
    async with _db_pool.acquire() as conn:
        await conn.execute(
            """UPDATE sessions
               SET messages = $2::jsonb, updated_at = now()
               WHERE session_id = $1""",
            session_id,
            json.dumps(history),
        )


# -- ENDPOINTS -----------------------------------------------------------------

class LoginRequest(BaseModel):
    password: str


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.post("/login")
async def login(req: LoginRequest):
    if req.password != ACCESS_PASSWORD:
        raise HTTPException(status_code=401, detail="Password errata")
    session_id = str(uuid.uuid4())
    async with _db_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO sessions (session_id) VALUES ($1)", session_id
        )
    return {"session_id": session_id}


SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}


async def _chat_generator(
    session_id: str,
    message: str,
    file_bytes: "Optional[bytes]",
    file_name: "Optional[str]",
    mime: "Optional[str]",
):
    """
    Async generator - invia Server-Sent Events al client.

    FRONTEND JS - sostituire il fetch classico con questo pattern:

        const formData = new FormData();
        formData.append('session_id', sessionId);
        formData.append('message', userMessage);
        // formData.append('file', fileInput.files[0]);  // opzionale

        const response = await fetch('/chat', { method: 'POST', body: formData });
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let botText = '';

        appendBotBubble('');  // crea bubble vuoto nel DOM

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const parts = buffer.split('\\n\\n');
            buffer = parts.pop();
            for (const part of parts) {
                if (!part.startsWith('data: ')) continue;
                const event = JSON.parse(part.slice(6));
                if (event.type === 'delta') {
                    botText += event.text;
                    updateLastBotBubble(botText);
                } else if (event.type === 'done') {
                    enableInput();
                } else if (event.type === 'error') {
                    showError(event.message);
                }
            }
        }

    NOTA: EventSource non supporta POST - usa fetch() con ReadableStream.
    """
    history = await _get_history(session_id)
    if history is None:
        yield f"data: {json.dumps({'type': 'error', 'message': 'Sessione non valida'})}\n\n"
        return

    # Web search - invariato
    contesto = ""
    if message.strip():
        try:
            risultati = tavily_client.search(message, max_results=3, search_depth="basic")
            for r in risultati["results"]:
                contesto += f"- {r['title']}: {r['content']}\n\n"
        except Exception:
            contesto = ""

    text_message = message
    if contesto:
        text_message = f"Contesto aggiornato dal web:\n{contesto}\n\nDomanda: {message}"

    # Costruisce i content block - invariato
    content_blocks: List[dict] = []

    if file_bytes and file_name:
        if mime in SUPPORTED_IMAGE_TYPES:
            b64 = base64.standard_b64encode(file_bytes).decode("utf-8")
            content_blocks.append({
                "type": "image",
                "source": {"type": "base64", "media_type": mime, "data": b64},
            })
        elif mime == "application/pdf":
            b64 = base64.standard_b64encode(file_bytes).decode("utf-8")
            content_blocks.append({
                "type": "document",
                "source": {"type": "base64", "media_type": "application/pdf", "data": b64},
            })
        else:
            try:
                file_text = file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                file_text = file_bytes.decode("latin-1")
            text_message = f"[File allegato: {file_name}]\n```\n{file_text}\n```\n\n{text_message}".strip()

    content_blocks.append({"type": "text", "text": text_message or "Analizza questo contenuto."})
    history.append({"role": "user", "content": content_blocks})

    # -- STREAMING -------------------------------------------------------------
    full_response = ""
    try:
        async with anthropic_client.messages.stream(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=history,
        ) as stream:
            async for text in stream.text_stream:
                full_response += text
                yield f"data: {json.dumps({'type': 'delta', 'text': text})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        return

    # Salva su PostgreSQL solo dopo che lo stream e' completato
    history.append({"role": "assistant", "content": full_response})
    await _save_history(session_id, history)

    yield f"data: {json.dumps({'type': 'done'})}\n\n"


@app.post("/chat")
async def chat(
    session_id: str = Form(...),
    message: str = Form(default=""),
    file: Optional[UploadFile] = File(default=None),
):
    # Il file va letto qui - UploadFile non e' accessibile dentro il generator asincrono
    file_bytes = None
    file_name = None
    mime = None
    if file and file.filename:
        file_bytes = await file.read()
        file_name = file.filename
        mime = file.content_type or ""

    # FRONTEND JS - questo endpoint ora ritorna SSE, non piu' JSON.
    return StreamingResponse(
        _chat_generator(session_id, message, file_bytes, file_name, mime),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    if _db_pool:
        async with _db_pool.acquire() as conn:
            await conn.execute(
                """UPDATE sessions
                   SET messages = '[]'::jsonb, updated_at = now()
                   WHERE session_id = $1""",
                session_id,
            )
    return {"status": "ok"}
