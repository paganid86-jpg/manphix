from fastapi import FastAPI, HTTPException, Form, UploadFile, File, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic
import groq as groq_sdk
from tavily import TavilyClient
import os
import secrets
import httpx
import asyncpg
import uuid
import base64
import time
import json
import re
import asyncio
import logging
from collections import defaultdict
from typing import List, Dict, Optional, Any, AsyncGenerator
from contextlib import asynccontextmanager

# --- LOGGING STRUTTURATO ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("manphix")

# --- CONFIGURAZIONE DATABASE ---
DATABASE_URL = os.environ.get("DATABASE_URL")

db_pool = None

async def init_db():
    """Crea le tabelle e gli indici se non esistono ancora"""
    async with db_pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id SERIAL PRIMARY KEY,
                session_id TEXT REFERENCES sessions(session_id),
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                id SERIAL PRIMARY KEY,
                session_id TEXT REFERENCES sessions(session_id),
                summary TEXT NOT NULL,
                message_count INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        # --- MIGLIORIA 1: indici per velocizzare le query per sessione ---
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_session_id
            ON messages(session_id)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_summaries_session_id
            ON summaries(session_id)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_summaries_created_at
            ON summaries(created_at DESC)
        """)

        # --- FASE 2 RAG: pgvector + knowledge_chunks ---
        # Prova ad abilitare pgvector (disponibile su Render PostgreSQL)
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS knowledge_chunks (
                    id         SERIAL PRIMARY KEY,
                    content    TEXT NOT NULL,
                    embedding  VECTOR({VOYAGE_DIMS}),
                    source     TEXT NOT NULL DEFAULT 'manual',
                    topic      TEXT[],
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            # Indice GIN per full-text search in italiano
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_kc_fts
                ON knowledge_chunks
                USING GIN (to_tsvector('italian', content))
            """)
            logger.info("pgvector abilitato e tabella knowledge_chunks pronta")
        except Exception as e:
            logger.warning(f"pgvector non disponibile — vector store disabilitato: {e}")

        logger.info("Database inizializzato con indici")

SESSION_EXPIRY_HOURS = 24  # sessioni inattive da più di 24h vengono eliminate

async def _session_cleanup_loop():
    """
    Job async che gira ogni ora ed elimina sessioni scadute (>24h senza messaggi).
    Prevenzione accumulo dati, tutela privacy.
    """
    while True:
        try:
            await asyncio.sleep(3600)  # ogni ora
            async with db_pool.acquire() as conn:
                # Elimina messaggi di sessioni scadute
                deleted = await conn.fetchval("""
                    WITH expired AS (
                        SELECT s.session_id
                        FROM sessions s
                        WHERE NOT EXISTS (
                            SELECT 1 FROM messages m
                            WHERE m.session_id = s.session_id
                              AND m.created_at > NOW() - INTERVAL '24 hours'
                        )
                        AND s.created_at < NOW() - INTERVAL '24 hours'
                    )
                    DELETE FROM sessions WHERE session_id IN (SELECT session_id FROM expired)
                    RETURNING session_id
                """)
                if deleted:
                    logger.info(f"Cleanup: rimossa sessione scaduta {deleted}")
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Session cleanup errore: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool
    db_pool = await asyncpg.create_pool(DATABASE_URL)
    await init_db()
    # Avvia cleanup job in background (non bloccante)
    cleanup_task = asyncio.create_task(_session_cleanup_loop())
    # Indicizza KB nel vector store se non già fatto (fire-and-forget)
    asyncio.create_task(index_kb_to_vector_store())
    yield
    cleanup_task.cancel()
    await db_pool.close()

app = FastAPI(lifespan=lifespan)

# --- CONFIGURAZIONE GITHUB ---
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO_OWNER = "paganid86-jpg"
REPO_NAME = "manphix-brain"
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# --- MIGLIORIA: confronto password sicuro (no timing attack) ---
ACCESS_PASSWORD = os.environ.get("MANPHIX_PASSWORD", "manphix2024")

# --- MIGLIORIA: KNOWLEDGE BASE CACHE con TTL 5 minuti ---
# Evita di fare fetch GitHub ad ogni singola richiesta chat
KB_CACHE_TTL = 300  # secondi
_kb_cache: Dict[str, Any] = {"main": None, "learnings": None}
_kb_cache_ts: Dict[str, float] = {"main": 0.0, "learnings": 0.0}
anthropic_client = anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
groq_client = groq_sdk.Groq(api_key=os.environ.get("GROQ_API_KEY")) if os.environ.get("GROQ_API_KEY") else None
tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

# --- VOYAGE AI (embeddings per vector store) ---
VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY")
VOYAGE_MODEL   = "voyage-3-lite"
VOYAGE_DIMS    = 512        # dimensioni vettore voyage-3-lite

# --- OPENROUTER ---
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

AVAILABLE_MODELS = {
    "haiku": {
        "provider": "anthropic",
        "model_id": "claude-haiku-4-5-20251001",
        "display_name": "Claude Haiku",
        "description": "Veloce e preciso (Anthropic)"
    },
    "llama-70b": {
        "provider": "groq",
        "model_id": "llama-3.3-70b-versatile",
        "display_name": "Llama 3.3 70B",
        "description": "Potente e versatile (Groq)"
    },
    "llama-8b": {
        "provider": "groq",
        "model_id": "llama-3.1-8b-instant",
        "display_name": "Llama 3.1 8B",
        "description": "Ultra veloce (Groq)"
    },
    "deepseek-r1": {
        "provider": "groq",
        "model_id": "deepseek-r1-distill-qwen-32b",
        "display_name": "DeepSeek R1",
        "description": "Reasoning avanzato (Groq)"
    },
    # --- OPENROUTER ---
    # Modelli gratuiti (:free) non richiedono crediti OpenRouter
    # Aggiungi qui altri modelli OpenRouter futuri (DeepSeek, Qwen, Llama, ecc.)
    "nemotron-super": {
        "provider": "openrouter",
        "model_id": "nvidia/nemotron-3-super-120b-a12b:free",
        "display_name": "Nemotron 3 Super 120B",
        "description": "NVIDIA 120B, 1M ctx — gratuito (OpenRouter)"
    },
    "nemotron-nano": {
        "provider": "openrouter",
        "model_id": "nvidia/nemotron-3-nano-30b-a3b:free",
        "display_name": "Nemotron 3 Nano 30B",
        "description": "NVIDIA 30B leggero, 128K ctx — gratuito (OpenRouter)"
    },
}

# --- MIGLIORIA 2: RATE LIMITING IN-MEMORY ---
# Max 20 richieste per sessione per finestra di 60 secondi
RATE_LIMIT_MAX = 20
RATE_LIMIT_WINDOW = 60  # secondi
_rate_store: Dict[str, List[float]] = defaultdict(list)

def check_rate_limit(session_id: str) -> bool:
    """
    Ritorna True se la richiesta è consentita, False se supera il limite.
    Implementazione sliding window.
    """
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW
    # Rimuove timestamp fuori dalla finestra
    _rate_store[session_id] = [t for t in _rate_store[session_id] if t > window_start]
    if len(_rate_store[session_id]) >= RATE_LIMIT_MAX:
        return False
    _rate_store[session_id].append(now)
    return True

# --- MIGLIORIA 3: VALIDAZIONE FILE SERVER-SIDE ---
FILE_SIZE_LIMIT = 10 * 1024 * 1024  # 10 MB

# Magic bytes per i tipi supportati
_MAGIC_BYTES: Dict[bytes, str] = {
    b"\xff\xd8\xff": "image/jpeg",
    b"\x89PNG":      "image/png",
    b"GIF8":         "image/gif",
    b"RIFF":         "image/webp",   # WebP: RIFF????WEBP
    b"%PDF":         "application/pdf",
}

SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
SUPPORTED_TEXT_TYPES  = {"text/plain", "text/markdown", "text/csv", "text/x-log",
                          "application/json", "text/x-python", "application/javascript"}

def validate_file(file_bytes: bytes, claimed_mime: str, filename: str) -> Optional[str]:
    """
    Valida dimensione e tipo del file.
    Ritorna None se OK, altrimenti il messaggio di errore.
    """
    if len(file_bytes) > FILE_SIZE_LIMIT:
        return f"File troppo grande ({len(file_bytes) // 1024 // 1024} MB). Limite: 10 MB."

    # Controlla magic bytes solo per i tipi binari dichiarati
    if claimed_mime in SUPPORTED_IMAGE_TYPES or claimed_mime == "application/pdf":
        matched = any(file_bytes[:len(magic)] == magic for magic in _MAGIC_BYTES)
        if not matched:
            return f"Il tipo del file non corrisponde al contenuto effettivo ({filename})."

    return None


async def stream_ai_response(
    messages: List[Dict],
    model_key: str,
    system_prompt: str,
) -> AsyncGenerator[str, None]:
    """Async generator that yields SSE-formatted strings."""
    cfg = AVAILABLE_MODELS.get(model_key, AVAILABLE_MODELS["haiku"])
    provider = cfg["provider"]
    model_id  = cfg["model_id"]
    t_start   = time.time()

    if provider == "anthropic":
        try:
            full_text  = ""
            tokens_in  = 0
            tokens_out = 0
            async with anthropic_client.messages.stream(
                model=model_id,
                max_tokens=1024,
                system=system_prompt,
                messages=messages,
            ) as stream:
                async for text_chunk in stream.text_stream:
                    full_text += text_chunk
                    payload = json.dumps({"type": "delta", "text": text_chunk})
                    yield f"data: {payload}\n\n"
                final_msg  = await stream.get_final_message()
                tokens_in  = final_msg.usage.input_tokens
                tokens_out = final_msg.usage.output_tokens
            latency_ms = int((time.time() - t_start) * 1000)
            usage_payload = json.dumps({
                "type":        "usage",
                "tokens_in":   tokens_in,
                "tokens_out":  tokens_out,
                "latency_ms":  latency_ms,
            })
            yield f"data: {usage_payload}\n\n"
        except anthropic.RateLimitError:
            logger.warning("Anthropic: quota esaurita")
            yield f'data: {json.dumps({"type": "error", "message": "Quota API Anthropic esaurita. Riprova tra qualche secondo."})}\n\n'
            return
        except anthropic.APITimeoutError:
            logger.error("Anthropic: timeout risposta")
            yield f'data: {json.dumps({"type": "error", "message": "Timeout risposta Anthropic. Riprova."})}\n\n'
            return
        except Exception as e:
            logger.error(f"Anthropic errore generico: {e}")
            yield f'data: {json.dumps({"type": "error", "message": str(e)})}\n\n'
            return

    elif provider == "groq":
        if not groq_client:
            yield f'data: {json.dumps({"type": "error", "message": "GROQ_API_KEY non configurata"})}\n\n'
            return
        try:
            groq_messages = [{"role": "system", "content": system_prompt}]
            for msg in messages:
                content = msg["content"]
                if isinstance(content, list):
                    text_parts = [b["text"] for b in content if b.get("type") == "text"]
                    content = "\n".join(text_parts) or "(contenuto non testuale)"
                groq_messages.append({"role": msg["role"], "content": content})

            loop = asyncio.get_event_loop()

            def _create_stream():
                return groq_client.chat.completions.create(
                    model=model_id,
                    messages=groq_messages,
                    max_tokens=1024,
                    stream=True,
                )

            stream_iter = await loop.run_in_executor(None, _create_stream)

            full_text  = ""
            tokens_in  = 0
            tokens_out = 0

            def _iter_chunks():
                nonlocal tokens_in, tokens_out
                for chunk in stream_iter:
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta:
                        yield delta
                    if chunk.usage:
                        tokens_in  = chunk.usage.prompt_tokens     or 0
                        tokens_out = chunk.usage.completion_tokens or 0

            for text_chunk in _iter_chunks():
                full_text += text_chunk
                payload = json.dumps({"type": "delta", "text": text_chunk})
                yield f"data: {payload}\n\n"

            latency_ms = int((time.time() - t_start) * 1000)
            usage_payload = json.dumps({
                "type":        "usage",
                "tokens_in":   tokens_in,
                "tokens_out":  tokens_out,
                "latency_ms":  latency_ms,
            })
            yield f"data: {usage_payload}\n\n"
        except Exception as e:
            logger.error(f"Groq errore: {e}")
            yield f'data: {json.dumps({"type": "error", "message": str(e)})}\n\n'
            return

    elif provider == "openrouter":
        # OpenRouter usa il formato OpenAI — compatibile con qualsiasi modello ospitato
        if not OPENROUTER_API_KEY:
            yield f'data: {json.dumps({"type": "error", "message": "OPENROUTER_API_KEY non configurata"})}\n\n'
            return
        try:
            # Costruisce la lista messaggi nel formato OpenAI (system come primo messaggio)
            or_messages = [{"role": "system", "content": system_prompt}]
            for msg in messages:
                content = msg["content"]
                # OpenRouter non supporta vision per tutti i modelli: flatten dei blocchi
                if isinstance(content, list):
                    text_parts = [b["text"] for b in content if b.get("type") == "text"]
                    content = "\n".join(text_parts) or "(contenuto non testuale)"
                or_messages.append({"role": msg["role"], "content": content})

            full_text  = ""
            tokens_in  = 0
            tokens_out = 0

            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST",
                    OPENROUTER_BASE_URL,
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json",
                        # Header consigliati da OpenRouter per il ranking e il routing
                        "HTTP-Referer": "https://manphix.onrender.com",
                        "X-Title": "Manphix",
                    },
                    json={
                        "model": model_id,
                        "messages": or_messages,
                        "stream": True,
                        "temperature": 0.7,
                        "max_tokens": 1024,
                    },
                ) as response:
                    if response.status_code != 200:
                        err = await response.aread()
                        logger.error(f"OpenRouter HTTP {response.status_code}: {err[:300]}")
                        yield f'data: {json.dumps({"type": "error", "message": f"OpenRouter errore HTTP {response.status_code}"})}\n\n'
                        return

                    # Parsa il flusso SSE riga per riga (formato OpenAI)
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        raw = line[6:].strip()
                        if raw == "[DONE]":
                            break
                        try:
                            chunk = json.loads(raw)
                            # Alcuni modelli mandano usage nell'ultimo chunk
                            if chunk.get("usage"):
                                tokens_in  = chunk["usage"].get("prompt_tokens", 0)
                                tokens_out = chunk["usage"].get("completion_tokens", 0)
                            delta      = (chunk.get("choices") or [{}])[0].get("delta", {})
                            text_chunk = delta.get("content") or ""
                            if text_chunk:
                                full_text += text_chunk
                                payload = json.dumps({"type": "delta", "text": text_chunk})
                                yield f"data: {payload}\n\n"
                        except json.JSONDecodeError:
                            continue

            latency_ms = int((time.time() - t_start) * 1000)
            usage_payload = json.dumps({
                "type":       "usage",
                "tokens_in":  tokens_in,
                "tokens_out": tokens_out,
                "latency_ms": latency_ms,
            })
            yield f"data: {usage_payload}\n\n"
        except httpx.TimeoutException:
            logger.error("OpenRouter: timeout risposta")
            yield f'data: {json.dumps({"type": "error", "message": "Timeout risposta OpenRouter. Riprova."})}\n\n'
            return
        except Exception as e:
            logger.error(f"OpenRouter errore: {e}")
            yield f'data: {json.dumps({"type": "error", "message": str(e)})}\n\n'
            return

    else:
        yield f'data: {json.dumps({"type": "error", "message": f"Provider sconosciuto: {provider}"})}\n\n'
        return

    yield 'data: {"type":"done"}\n\n'


SYSTEM_PROMPT = """Sei Manphix, un assistente informativo autorevole, pacato e cordiale, con un pizzico
di sarcasmo e humor giovanile. Non sei un chatbot generico — sei uno specialista con
una voce riconoscibile e un punto di vista proprio.

══════════════════════════════════════════════════════════
REGOLE ANTI-ALLUCINAZIONE — PRIORITÀ ASSOLUTA
══════════════════════════════════════════════════════════

1. FONTI PRIMA DI TUTTO
   Rispondi basandoti esclusivamente su:
   a) Risultati web forniti nel CONTESTO WEB (massima priorità)
   b) Knowledge Base e learnings forniti nella sezione apposita
   c) Fatti storici consolidati e non contestabili (matematica, fisica nota)
   La tua conoscenza interna è un fallback di ULTIMA istanza, mai la fonte primaria
   per eventi, statistiche o notizie recenti.

2. VIETATO INVENTARE
   Non inventare mai: statistiche, risultati sportivi, date, nomi, prezzi,
   citazioni, titoli di articoli, URL, dichiarazioni di persone.
   Se non hai una fonte verificabile, non affermare il dato.

3. FALLBACK ESPLICITO
   Se il contesto web non copre la domanda, dì letteralmente:
   "Non ho fonti aggiornate su questo. Vuoi che provi con una ricerca diversa?"
   Non riempire il vuoto con supposizioni.

4. FONTI CONTRADDITTORIE
   Se le fonti non concordano, mostralo chiaramente:
   "Le fonti si contraddicono: [fonte A dice X] vs [fonte B dice Y]. Preferisco la
   più recente / più affidabile perché..."

5. CITAZIONI INLINE
   Ogni volta che usi un'informazione proveniente da una fonte web, citala così:
   📌 [Nome fonte](url)
   Mettila alla fine della frase o del paragrafo che la usa.

6. DATI TEMPORALI
   Preferisci sempre la fonte più recente per fatti mutevoli (classifiche, prezzi,
   dichiarazioni). Se la fonte ha più di 6 mesi e riguarda un dato variabile,
   segnalalo: "(dato del [data], verifica se aggiornato)"

══════════════════════════════════════════════════════════

USA LA TUA CONOSCENZA EVOLUTIVA:
In ogni risposta, tieni conto dei dati forniti nella sezione 'KNOWLEDGE BASE ESTERNA'.
Questi dati rappresentano ciò che hai imparato dai tuoi errori o approfondimenti passati.

USA LA TUA MEMORIA STORICA:
Nella sezione 'MEMORIA CONVERSAZIONI PASSATE' trovi riassunti di conversazioni precedenti.
Usali per mantenere continuità, ricordare preferenze e riferimenti già discussi.

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
- Rispondi in italiano o inglese a seconda della lingua dell'utente.

STILE DI OUTPUT:
- Produci tabelle comparative quando si confrontano dati o opzioni.
- Offri analisi approfondite quando richiesto.
- Esprimi opinioni e commenti personali con tono diretto e schietto,
  distinguendoli chiaramente dai fatti (usa "secondo me" / "a mio avviso").
- Usa il sarcasmo e l'humor con intelligenza — mai volgare, sempre pertinente.

IDENTITÀ:
Sei Manphix. Non sei Claude, non sei ChatGPT, non sei un assistente generico.
Se ti chiedono chi sei, descrivi te stesso come Manphix senza menzionare
il modello AI sottostante."""


class LoginRequest(BaseModel):
    password: str


# --- FUNZIONI DB ---

async def create_session(session_id: str):
    async with db_pool.acquire() as conn:
        await conn.execute("INSERT INTO sessions (session_id) VALUES ($1)", session_id)

async def session_exists(session_id: str) -> bool:
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow("SELECT session_id FROM sessions WHERE session_id = $1", session_id)
        return row is not None

async def save_message(session_id: str, role: str, content: str):
    async with db_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES ($1, $2, $3)",
            session_id, role, content
        )

async def get_history(session_id: str) -> List[Dict]:
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT role, content FROM messages WHERE session_id = $1 ORDER BY created_at ASC",
            session_id
        )
        return [{"role": r["role"], "content": r["content"]} for r in rows]

async def count_messages(session_id: str) -> int:
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT COUNT(*) as cnt FROM messages WHERE session_id = $1", session_id
        )
        return row["cnt"]

async def save_summary(session_id: str, summary: str, message_count: int):
    async with db_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO summaries (session_id, summary, message_count) VALUES ($1, $2, $3)",
            session_id, summary, message_count
        )

# --- MIGLIORIA 5: generazione summaries asincrona e non-bloccante ---
async def generate_and_save_summary(session_id: str, history: List[Dict], message_count: int):
    """
    Livello 2: genera e salva riassunto ogni 7 messaggi.
    Usa il client async (non più sync) — viene eseguita come fire-and-forget
    tramite asyncio.create_task() per non bloccare lo streaming.
    """
    try:
        conversation_text = "\n".join([
            f"{m['role'].upper()}: {m['content'][:500]}"
            for m in history
        ])
        response = await anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": f"""Riassumi questa conversazione in modo conciso (max 150 parole).
Includi: argomenti trattati, opinioni espresse, informazioni importanti emerse.
Scrivi in italiano, in terza persona (es. "L'utente ha chiesto...").

CONVERSAZIONE:
{conversation_text}"""
            }]
        )
        summary = response.content[0].text
        await save_summary(session_id, summary, message_count)
        logger.info(f"Riassunto salvato per sessione {session_id} ({message_count} messaggi)")
    except Exception as e:
        logger.error(f"Errore generazione riassunto sessione {session_id}: {e}")

async def get_recent_summaries(limit: int = 5) -> str:
    """
    LIVELLO 3: recupera gli ultimi N riassunti da TUTTE le sessioni,
    ordinati dal più recente.
    """
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT summary, created_at
               FROM summaries
               ORDER BY created_at DESC
               LIMIT $1""",
            limit
        )
        if not rows:
            return ""
        summaries_text = ""
        for i, row in enumerate(reversed(rows), 1):
            date_str = row["created_at"].strftime("%d/%m/%Y %H:%M")
            summaries_text += f"\n[Conversazione {i} — {date_str}]\n{row['summary']}\n"
        return summaries_text

async def clear_session_messages(session_id: str):
    async with db_pool.acquire() as conn:
        await conn.execute("DELETE FROM messages WHERE session_id = $1", session_id)


# --- FUNZIONI DI RECUPERO CONOSCENZA ---

async def _fetch_github_file_content(file_path: str) -> str:
    """Fetch raw da GitHub, senza cache."""
    url = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/main/{file_path}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, headers=HEADERS)
            if resp.status_code == 200:
                return resp.text
            logger.warning(f"GitHub file {file_path}: HTTP {resp.status_code}")
            return ""
    except httpx.TimeoutException:
        logger.warning(f"GitHub file {file_path}: timeout")
        return ""
    except Exception as e:
        logger.error(f"GitHub file {file_path}: {e}")
        return ""

async def get_github_file_content(file_path: str) -> str:
    """
    Wrapper con cache TTL 5 min.
    Usa il valore in cache se fresco, altrimenti ri-fetcha e aggiorna.
    In caso di errore fetch, restituisce il valore cache precedente (stale) se disponibile.
    """
    now = time.time()
    if now - _kb_cache_ts["main"] < KB_CACHE_TTL and _kb_cache["main"] is not None:
        return _kb_cache["main"]
    fresh = await _fetch_github_file_content(file_path)
    if fresh:
        _kb_cache["main"] = fresh
        _kb_cache_ts["main"] = now
        logger.info("Knowledge Base (main) aggiornata dalla cache")
    elif _kb_cache["main"] is not None:
        logger.warning("GitHub main KB non raggiungibile — uso cache stale")
        return _kb_cache["main"]
    return fresh

async def _fetch_all_learnings() -> str:
    """Fetch raw learnings da GitHub, senza cache."""
    api_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/.learnings"
    all_lessons = ""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(api_url, headers=HEADERS)
            if resp.status_code != 200:
                logger.warning(f"GitHub learnings: HTTP {resp.status_code}")
                return ""
            files = resp.json()
            for file_info in files:
                if file_info["type"] == "file":
                    try:
                        f_resp = await client.get(file_info["download_url"], timeout=10.0)
                        if f_resp.status_code == 200:
                            all_lessons += f"\n--- Fonte: {file_info['name']} ---\n{f_resp.text}\n"
                    except Exception as fe:
                        logger.warning(f"GitHub learning file {file_info['name']}: {fe}")
            return all_lessons
    except httpx.TimeoutException:
        logger.warning("GitHub learnings: timeout")
        return ""
    except Exception as e:
        logger.error(f"GitHub learnings: {e}")
        return ""

async def get_all_learnings() -> str:
    """
    Wrapper con cache TTL 5 min per i learnings.
    Stessa logica stale-fallback di get_github_file_content.
    """
    now = time.time()
    if now - _kb_cache_ts["learnings"] < KB_CACHE_TTL and _kb_cache["learnings"] is not None:
        return _kb_cache["learnings"]
    fresh = await _fetch_all_learnings()
    if fresh:
        _kb_cache["learnings"] = fresh
        _kb_cache_ts["learnings"] = now
        logger.info("Knowledge Base (learnings) aggiornata dalla cache")
    elif _kb_cache["learnings"] is not None:
        logger.warning("GitHub learnings non raggiungibili — uso cache stale")
        return _kb_cache["learnings"]
    return fresh


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 RAG — VECTOR STORE + HYBRID SEARCH
# ══════════════════════════════════════════════════════════════════════════════

async def get_embedding(text: str) -> Optional[List[float]]:
    """Chiama Voyage AI per ottenere il vettore di embedding. Ritorna None su errore."""
    if not VOYAGE_API_KEY or not text.strip():
        return None
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "https://api.voyageai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {VOYAGE_API_KEY}"},
                json={"input": [text[:2000]], "model": VOYAGE_MODEL},
            )
            if resp.status_code == 200:
                return resp.json()["data"][0]["embedding"]
            logger.warning(f"Voyage AI: HTTP {resp.status_code} — {resp.text[:200]}")
            return None
    except Exception as e:
        logger.error(f"Voyage AI embedding errore: {e}")
        return None


def chunk_text(text: str, source: str, max_chars: int = 500) -> List[Dict]:
    """
    Divide un testo markdown in chunk semantici rispettando le sezioni.
    Ogni chunk mantiene il contesto dell'header di appartenenza.
    """
    chunks = []
    sections = re.split(r'\n(?=#{1,3} )', text)
    for section in sections:
        section = section.strip()
        if not section or len(section) < 30:
            continue
        if len(section) <= max_chars:
            chunks.append({"content": section, "source": source})
        else:
            # Sezione lunga: split per paragrafi
            paragraphs = section.split("\n\n")
            current = ""
            for p in paragraphs:
                if len(current) + len(p) + 2 > max_chars and current:
                    chunks.append({"content": current.strip(), "source": source})
                    current = p
                else:
                    current = (current + "\n\n" + p).strip() if current else p
            if current.strip() and len(current.strip()) >= 30:
                chunks.append({"content": current.strip(), "source": source})
    return chunks


async def index_kb_to_vector_store(force: bool = False):
    """
    Fetch KB da GitHub, split in chunks, embed con Voyage AI, salva in pgvector.
    force=True: cancella i chunk esistenti (esclusi 'conversation') e re-indicizza.
    force=False: salta se già indicizzato.
    """
    if not VOYAGE_API_KEY:
        logger.info("VOYAGE_API_KEY non impostata — KB indexing saltato")
        return

    try:
        async with db_pool.acquire() as conn:
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM knowledge_chunks WHERE source != 'conversation'"
            )
            if count > 0 and not force:
                logger.info(f"Vector store già popolato ({count} chunks KB) — skip indicizzazione")
                return
            if force:
                await conn.execute(
                    "DELETE FROM knowledge_chunks WHERE source != 'conversation'"
                )
                logger.info("Vector store: chunk KB rimossi per re-indicizzazione")

        # Fetch raw da GitHub (bypass cache per avere contenuto fresco)
        kb_main      = await _fetch_github_file_content("CLAUDE.md")
        kb_learnings = await _fetch_all_learnings()

        all_chunks: List[Dict] = []
        if kb_main:
            all_chunks.extend(chunk_text(kb_main, "dante.md"))
        if kb_learnings:
            for section in kb_learnings.split("--- Fonte:"):
                section = section.strip()
                if not section:
                    continue
                sep = section.find(" ---\n")
                if sep > 0:
                    fname   = section[:sep].strip()
                    content = section[sep + 4:].strip()
                else:
                    fname, content = "learnings", section
                all_chunks.extend(chunk_text(content, f"learnings/{fname}"))

        if not all_chunks:
            logger.warning("Vector store: nessun chunk da indicizzare — KB vuota o non raggiungibile")
            return

        # Embed e salva in batch da 10 per rispettare i limiti Voyage
        saved = 0
        for i in range(0, len(all_chunks), 10):
            batch = all_chunks[i:i + 10]
            texts = [c["content"] for c in batch]
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.post(
                        "https://api.voyageai.com/v1/embeddings",
                        headers={"Authorization": f"Bearer {VOYAGE_API_KEY}"},
                        json={"input": texts, "model": VOYAGE_MODEL},
                    )
                if resp.status_code != 200:
                    logger.warning(f"Voyage batch {i}: HTTP {resp.status_code}")
                    continue
                embeddings = resp.json()["data"]
            except Exception as e:
                logger.error(f"Voyage batch {i} errore: {e}")
                continue

            async with db_pool.acquire() as conn:
                for j, chunk in enumerate(batch):
                    emb     = embeddings[j]["embedding"]
                    emb_str = "[" + ",".join(str(x) for x in emb) + "]"
                    await conn.execute(
                        """INSERT INTO knowledge_chunks (content, embedding, source)
                           VALUES ($1, $2::vector, $3)""",
                        chunk["content"], emb_str, chunk["source"]
                    )
                    saved += 1

        logger.info(f"Vector store: {saved}/{len(all_chunks)} chunks indicizzati")

    except Exception as e:
        logger.error(f"index_kb_to_vector_store errore: {e}")


def rrf_merge(vec_results: list, kw_results: list, k: int = 60) -> list:
    """Reciprocal Rank Fusion: fonde risultati vector search e keyword search."""
    scores:   Dict[int, float] = {}
    all_rows: Dict[int, dict]  = {}

    for rank, row in enumerate(vec_results):
        rid = row["id"]
        scores[rid]   = scores.get(rid, 0.0) + 1.0 / (k + rank + 1)
        all_rows[rid] = dict(row)

    for rank, row in enumerate(kw_results):
        rid = row["id"]
        scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + rank + 1)
        if rid not in all_rows:
            all_rows[rid] = dict(row)

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [all_rows[rid] for rid in sorted_ids]


async def hybrid_search(query: str, k: int = 6) -> List[Dict]:
    """
    Ricerca ibrida: vector similarity (Voyage AI) + full-text PostgreSQL, fusa con RRF.
    Ritorna lista di dict {content, source}. Lista vuota se non disponibile.
    """
    if not VOYAGE_API_KEY:
        return []
    try:
        q_emb = await get_embedding(query)
        if not q_emb:
            return []
        q_emb_str = "[" + ",".join(str(x) for x in q_emb) + "]"

        async with db_pool.acquire() as conn:
            # 1. Vector search — top k per cosine similarity
            vec_rows = await conn.fetch("""
                SELECT id, content, source,
                       1 - (embedding <=> $1::vector) AS score
                FROM knowledge_chunks
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> $1::vector
                LIMIT $2
            """, q_emb_str, k)

            # 2. Full-text search — dizionario italiano
            kw_rows = await conn.fetch("""
                SELECT id, content, source,
                       ts_rank(
                           to_tsvector('italian', content),
                           plainto_tsquery('italian', $1)
                       ) AS score
                FROM knowledge_chunks
                WHERE to_tsvector('italian', content)
                      @@ plainto_tsquery('italian', $1)
                ORDER BY score DESC
                LIMIT $2
            """, query, k)

        merged = rrf_merge(list(vec_rows), list(kw_rows))
        logger.info(
            f"Hybrid search: {len(vec_rows)} vector + {len(kw_rows)} keyword → "
            f"{len(merged)} merged per '{query[:60]}'"
        )
        return merged[:k]

    except Exception as e:
        logger.error(f"Hybrid search errore: {e}")
        return []


async def extract_and_store_learnings(conversation: List[Dict]):
    """
    Fire-and-forget: estrae fatti nuovi su Dante dalla conversazione appena conclusa
    e li salva nel vector store come memoria permanente.
    Richiede entrambe le API (Anthropic + Voyage). Si disattiva silenziosamente
    se una delle due non è disponibile o la quota è esaurita.
    """
    if not VOYAGE_API_KEY:
        return
    try:
        recent    = conversation[-8:]
        conv_text = "\n".join([
            f"{m['role'].upper()}: {str(m['content'])[:300]}"
            for m in recent
        ])
        response = await anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            messages=[{"role": "user", "content": f"""Analizza questa conversazione ed estrai SOLO fatti nuovi e verificabili
su Dante (preferenze, abitudini, eventi della sua vita, opinioni esplicitamente dichiarate).
Regole: solo affermazioni esplicite di Dante, no inferenze, no speculazioni.
Se non ci sono fatti nuovi, rispondi esattamente con: []

Output JSON (array, max 3 elementi):
[{{"fatto": "...", "topic": ["tag"]}}]

CONVERSAZIONE:
{conv_text}"""}]
        )
        text  = response.content[0].text.strip()
        match = re.search(r'\[.*?\]', text, re.DOTALL)
        if not match:
            return
        facts = json.loads(match.group())
        if not isinstance(facts, list) or not facts:
            return

        saved = 0
        for fact in facts:
            fatto = fact.get("fatto", "").strip()
            if not fatto:
                continue
            emb = await get_embedding(fatto)
            if not emb:
                continue
            emb_str = "[" + ",".join(str(x) for x in emb) + "]"
            async with db_pool.acquire() as conn:
                await conn.execute(
                    """INSERT INTO knowledge_chunks (content, embedding, source, topic)
                       VALUES ($1, $2::vector, 'conversation', $3)""",
                    fatto, emb_str, fact.get("topic", [])
                )
            saved += 1

        if saved:
            logger.info(f"Conversational learning: {saved} fatti nuovi salvati nel vector store")

    except anthropic.RateLimitError:
        logger.warning("Conversational learning: quota Anthropic esaurita — skip")
    except Exception as e:
        logger.error(f"Conversational learning errore: {e}")


# --- ENDPOINTS ---

@app.get("/")
async def root():
    return FileResponse("static/index.html")

# --- MIGLIORIA: Health check per Render monitoring ---
@app.get("/health")
async def health():
    """Endpoint leggero per uptime monitoring.
    Verifica anche connettività DB (ping semplice)."""
    db_ok = False
    try:
        async with db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        db_ok = True
    except Exception as e:
        logger.error(f"Health check DB fallito: {e}")
    status = "ok" if db_ok else "degraded"
    return JSONResponse(
        content={"status": status, "db": db_ok},
        status_code=200 if db_ok else 503,
    )

@app.get("/api/models")
async def get_models():
    """Restituisce i modelli disponibili in base alle API key configurate."""
    provider_enabled = {
        "anthropic":  bool(os.environ.get("ANTHROPIC_API_KEY")),
        "groq":       bool(os.environ.get("GROQ_API_KEY")),
        "openrouter": bool(OPENROUTER_API_KEY),
    }
    available = [
        {
            "key":          key,
            "display_name": cfg["display_name"],
            "description":  cfg["description"],
            "provider":     cfg["provider"],
        }
        for key, cfg in AVAILABLE_MODELS.items()
        if provider_enabled.get(cfg["provider"], False)
    ]
    return {"models": available}

@app.post("/login")
async def login(req: LoginRequest):
    # compare_digest previene timing attacks (confronto a tempo costante)
    if not secrets.compare_digest(req.password, ACCESS_PASSWORD):
        raise HTTPException(status_code=401, detail="Password errata")
    session_id = str(uuid.uuid4())
    await create_session(session_id)
    logger.info(f"Nuova sessione creata: {session_id}")
    return {"session_id": session_id}


@app.post("/chat")
async def chat(
    session_id: str = Form(...),
    message: str = Form(default=""),
    model: str = Form(default="haiku"),
    file: Optional[UploadFile] = File(default=None),
):
    if not await session_exists(session_id):
        raise HTTPException(status_code=401, detail="Sessione non valida")

    # --- MIGLIORIA 2: controllo rate limit ---
    if not check_rate_limit(session_id):
        logger.warning(f"Rate limit superato per sessione {session_id}")
        raise HTTPException(
            status_code=429,
            detail=f"Troppe richieste. Limite: {RATE_LIMIT_MAX} messaggi al minuto."
        )

    # Legge il file PRIMA di entrare nel generator (UploadFile non è leggibile dentro)
    file_bytes = None
    file_filename = None
    file_mime = None
    if file and file.filename:
        file_bytes    = await file.read()
        file_filename = file.filename
        file_mime     = file.content_type or ""

        # --- MIGLIORIA 3: validazione file server-side ---
        error_msg = validate_file(file_bytes, file_mime, file_filename)
        if error_msg:
            raise HTTPException(status_code=400, detail=error_msg)

    async def _chat_generator() -> AsyncGenerator[str, None]:
        # 1. Profilo personale: i learnings (dante.md + altri) sono SEMPRE caricati.
        #    Sono la "memoria permanente" di Manphix su Dante — piccoli e sempre rilevanti.
        personal_profile = await get_all_learnings()

        # 2. Contesto aggiuntivo via hybrid search (solo chunk rilevanti alla query)
        relevant_chunks = await hybrid_search(message) if message.strip() else []
        if relevant_chunks:
            chunk_context = "\n\n".join(
                f"[{c['source']}] {c['content']}" for c in relevant_chunks
            )
            kb_context = f"{personal_profile}\n\n### CONTESTO RILEVANTE ALLA QUERY:\n{chunk_context}"
        else:
            # Fallback: carica anche CLAUDE.md dal repo brain
            knowledge_main = await get_github_file_content("CLAUDE.md")
            kb_context     = f"{knowledge_main}\n{personal_profile}"

        # 2. LIVELLO 3: recupero riassunti delle conversazioni passate
        past_summaries = await get_recent_summaries(limit=5)

        # 3. Recupero storia della sessione corrente dal DB
        history = await get_history(session_id)

        # 4. Query preprocessing + ricerca web (solo se c'è testo)
        text_message = message
        if message.strip():
            # --- MIGLIORIA 4: logging errori query preprocessing ---
            try:
                query_response = await anthropic_client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=100,
                    messages=[{
                        "role": "user",
                        "content": f"""Trasforma questa domanda in una query di ricerca web ottimale.
Rispondi SOLO con la query, niente altro. Max 10 parole. In italiano o inglese a seconda della lingua.
Aggiungi l'anno corrente (2026) se la domanda riguarda eventi recenti o notizie.

Domanda: {message}"""
                    }]
                )
                search_query = query_response.content[0].text.strip()
            except anthropic.RateLimitError:
                logger.warning("Anthropic query preprocessing: quota esaurita, uso messaggio originale")
                search_query = message
            except Exception as e:
                logger.warning(f"Anthropic query preprocessing: {e}, uso messaggio originale")
                search_query = message

            try:
                risultati = tavily_client.search(
                    search_query,
                    max_results=5,
                    search_depth="advanced",
                    topic="news",
                    include_answer=True,
                    include_raw_content=False
                )

                # Risposta sintetizzata da Tavily (quando disponibile)
                tavily_answer = risultati.get("answer", "")

                # Costruisce le righe fonte con URL e data (se disponibile)
                source_lines = []
                for r in risultati["results"]:
                    url      = r.get("url", "")
                    date_raw = r.get("published_date", "")
                    date_str = f" [{date_raw[:10]}]" if date_raw else ""
                    snippet  = r["content"][:400].replace("\n", " ")
                    source_lines.append(f"• [{r['title']}]({url}){date_str}: {snippet}")

                contesto = "\n".join(source_lines)

                if tavily_answer:
                    contesto = f"Risposta diretta: {tavily_answer}\n\nFonti di dettaglio:\n{contesto}"

                if contesto and len(contesto.strip()) > 50:
                    text_message = (
                        f"CONTESTO WEB (basa la risposta su queste fonti, non sulla tua conoscenza interna):\n"
                        f"{contesto}\n\n"
                        f"Regola citazioni: cita ogni fatto usato con 📌 [titolo](url) inline.\n"
                        f"Se le fonti non coprono la domanda, dillo esplicitamente.\n\n"
                        f"Domanda: {message}"
                    )
            except Exception as e:
                logger.warning(f"Tavily search fallita ('{search_query}'): {e}")
                # Notifica il frontend che la ricerca web non era disponibile
                yield f'data: {json.dumps({"type": "search_skipped", "reason": "Web search non disponibile — rispondo con conoscenza locale."})}\n\n'

        # 5. Gestione file allegato
        db_message      = text_message
        message_content = text_message

        if file_bytes is not None and file_filename:
            content_blocks: List[dict] = []
            if file_mime in SUPPORTED_IMAGE_TYPES:
                b64 = base64.standard_b64encode(file_bytes).decode("utf-8")
                content_blocks.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": file_mime, "data": b64},
                })
                db_message = f"[Immagine allegata]\n{text_message}".strip()
            elif file_mime == "application/pdf":
                b64 = base64.standard_b64encode(file_bytes).decode("utf-8")
                content_blocks.append({
                    "type": "document",
                    "source": {"type": "base64", "media_type": "application/pdf", "data": b64},
                })
                db_message = f"[PDF allegato: {file_filename}]\n{text_message}".strip()
            else:
                try:
                    file_text = file_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    file_text = file_bytes.decode("latin-1")
                text_message = f"[File: {file_filename}]\n```\n{file_text}\n```\n\n{text_message}".strip()
                db_message   = text_message
            content_blocks.append({"type": "text", "text": text_message or "Analizza questo contenuto."})
            message_content = content_blocks

        # 6. Salva messaggio utente nel DB e aggiungi alla history
        await save_message(session_id, "user", db_message)
        history.append({"role": "user", "content": message_content})

        # 7. Costruisci prompt arricchito con tutti e 3 i livelli di memoria
        enriched_prompt = SYSTEM_PROMPT
        enriched_prompt += f"\n\n### KNOWLEDGE BASE ESTERNA (I tuoi apprendimenti):\n{kb_context}"
        if past_summaries:
            enriched_prompt += f"\n\n### MEMORIA CONVERSAZIONI PASSATE:\n{past_summaries}"

        # 8. Stream risposta AI
        full_response = ""
        async for sse_chunk in stream_ai_response(history, model, enriched_prompt):
            if sse_chunk.startswith("data: "):
                try:
                    payload = json.loads(sse_chunk[6:].strip())
                    if payload.get("type") == "delta":
                        full_response += payload.get("text", "")
                except Exception:
                    pass
            yield sse_chunk

        # 9. Salva risposta di Manphix nel DB
        if full_response:
            await save_message(session_id, "assistant", full_response)

            # 10. Ogni 7 messaggi genera riassunto in background (non bloccante)
            # --- MIGLIORIA 5: asyncio.create_task() — fire-and-forget ---
            total_messages = await count_messages(session_id)
            history.append({"role": "assistant", "content": full_response})
            if total_messages % 7 == 0:
                asyncio.create_task(
                    generate_and_save_summary(session_id, history, total_messages)
                )
            # 11. Conversational learning: estrae fatti nuovi e aggiorna il vector store
            #     Richiede Anthropic billing — si disattiva silenziosamente se esaurito
            asyncio.create_task(extract_and_store_learnings(history))

    return StreamingResponse(
        _chat_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/admin/reindex")
async def reindex_kb(request: Request):
    """
    Forza re-indicizzazione completa della KB nel vector store.
    Cancella i chunk esistenti (esclusi 'conversation') e re-indica da GitHub.
    Protetto dalla stessa password di accesso.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Body JSON richiesto")
    if not secrets.compare_digest(body.get("password", ""), ACCESS_PASSWORD):
        raise HTTPException(status_code=401, detail="Non autorizzato")
    asyncio.create_task(index_kb_to_vector_store(force=True))
    return {"status": "re-indicizzazione avviata in background"}


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    await clear_session_messages(session_id)
    logger.info(f"Sessione {session_id} cancellata")
    return {"status": "ok"}
