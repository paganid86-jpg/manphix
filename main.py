from fastapi import FastAPI, HTTPException, Form, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic
from tavily import TavilyClient
import os
import httpx
import asyncpg
import uuid
import base64
from typing import List, Dict, Optional
from contextlib import asynccontextmanager

# --- CONFIGURAZIONE DATABASE ---
DATABASE_URL = os.environ.get("DATABASE_URL")

db_pool = None

async def init_db():
    """Crea le tabelle se non esistono ancora"""
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool
    db_pool = await asyncpg.create_pool(DATABASE_URL)
    await init_db()
    yield
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

ACCESS_PASSWORD = os.environ.get("MANPHIX_PASSWORD", "manphix2024")
anthropic_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

SYSTEM_PROMPT = """Sei Manphix, un assistente informativo autorevole, pacato e cordiale, con un pizzico 
di sarcasmo e humor giovanile. Non sei un chatbot generico — sei uno specialista con 
una voce riconoscibile e un punto di vista proprio.

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

SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}

class LoginRequest(BaseModel):
    password: str

class ChatResponse(BaseModel):
    response: str

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

async def generate_and_save_summary(session_id: str, history: List[Dict], message_count: int):
    """Livello 2: genera e salva riassunto ogni 7 messaggi"""
    try:
        conversation_text = "\n".join([
            f"{m['role'].upper()}: {m['content'][:500]}"
            for m in history
        ])
        response = anthropic_client.messages.create(
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
        print(f"[Manphix] Riassunto salvato per sessione {session_id} dopo {message_count} messaggi")
    except Exception as e:
        print(f"[Manphix] Errore generazione riassunto: {e}")

async def get_recent_summaries(limit: int = 5) -> str:
    """
    LIVELLO 3: recupera gli ultimi N riassunti da TUTTE le sessioni,
    ordinati dal più recente. Servono per dare a Manphix memoria storica
    delle conversazioni passate, indipendentemente dalla sessione corrente.
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
        
        # Costruiamo il testo da iniettare nel prompt, dal più vecchio al più recente
        summaries_text = ""
        for i, row in enumerate(reversed(rows), 1):
            date_str = row["created_at"].strftime("%d/%m/%Y %H:%M")
            summaries_text += f"\n[Conversazione {i} — {date_str}]\n{row['summary']}\n"
        
        return summaries_text

async def clear_session_messages(session_id: str):
    async with db_pool.acquire() as conn:
        await conn.execute("DELETE FROM messages WHERE session_id = $1", session_id)

# --- FUNZIONI DI RECUPERO CONOSCENZA ---

async def get_github_file_content(file_path: str):
    url = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/main/{file_path}"
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(url, headers=HEADERS)
            return resp.text if resp.status_code == 200 else ""
        except:
            return ""

async def get_all_learnings():
    api_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/.learnings"
    all_lessons = ""
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(api_url, headers=HEADERS)
            if resp.status_code == 200:
                files = resp.json()
                for file_info in files:
                    if file_info["type"] == "file":
                        f_resp = await client.get(file_info["download_url"])
                        if f_resp.status_code == 200:
                            all_lessons += f"\n--- Fonte: {file_info['name']} ---\n{f_resp.text}\n"
            return all_lessons
        except:
            return ""

# --- ENDPOINTS ---

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.post("/login")
async def login(req: LoginRequest):
    if req.password != ACCESS_PASSWORD:
        raise HTTPException(status_code=401, detail="Password errata")
    session_id = str(uuid.uuid4())
    await create_session(session_id)
    return {"session_id": session_id}

@app.post("/chat", response_model=ChatResponse)
async def chat(
    session_id: str = Form(...),
    message: str = Form(default=""),
    file: Optional[UploadFile] = File(default=None),
):
    if not await session_exists(session_id):
        raise HTTPException(status_code=401, detail="Sessione non valida")

    # 1. Recupero conoscenza da GitHub
    knowledge_main = await get_github_file_content("CLAUDE.md")
    knowledge_folder = await get_all_learnings()
    full_external_knowledge = f"{knowledge_main}\n{knowledge_folder}"

    # 2. LIVELLO 3: recupero riassunti delle conversazioni passate
    past_summaries = await get_recent_summaries(limit=5)

    # 3. Recupero storia della sessione corrente dal DB
    history = await get_history(session_id)

    # 4. Query preprocessing + ricerca web (solo se c'è testo)
    text_message = message
    if message.strip():
        try:
            query_response = anthropic_client.messages.create(
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
        except:
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
            contesto = "".join([f"- {r['title']}: {r['content']}\n\n" for r in risultati["results"]])
            if contesto and len(contesto.strip()) > 50:
                text_message = f"Contesto Web (usa solo se pertinente alla domanda):\n{contesto}\n\nDomanda: {message}"
        except:
            pass

    # 5. Gestione file allegato — costruisce content blocks e testo per il DB
    content_blocks: List[dict] = []
    db_message = text_message  # versione testo-only per il database

    if file and file.filename:
        file_bytes = await file.read()
        mime = file.content_type or ""

        if mime in SUPPORTED_IMAGE_TYPES:
            b64 = base64.standard_b64encode(file_bytes).decode("utf-8")
            content_blocks.append({
                "type": "image",
                "source": {"type": "base64", "media_type": mime, "data": b64},
            })
            db_message = f"[Immagine allegata]\n{text_message}".strip()
        elif mime == "application/pdf":
            b64 = base64.standard_b64encode(file_bytes).decode("utf-8")
            content_blocks.append({
                "type": "document",
                "source": {"type": "base64", "media_type": "application/pdf", "data": b64},
            })
            db_message = f"[PDF allegato: {file.filename}]\n{text_message}".strip()
        else:
            try:
                file_text = file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                file_text = file_bytes.decode("latin-1")
            text_message = f"[File: {file.filename}]\n```\n{file_text}\n```\n\n{text_message}".strip()
            db_message = text_message

    content_blocks.append({"type": "text", "text": text_message or "Analizza questo contenuto."})

    # 6. Salva nel DB (testo, senza base64) e aggiungi alla history per la chiamata API
    await save_message(session_id, "user", db_message)
    history.append({"role": "user", "content": content_blocks})

    # 7. Costruisci prompt arricchito con tutti e 3 i livelli di memoria
    enriched_prompt = SYSTEM_PROMPT
    enriched_prompt += f"\n\n### KNOWLEDGE BASE ESTERNA (I tuoi apprendimenti):\n{full_external_knowledge}"
    if past_summaries:
        enriched_prompt += f"\n\n### MEMORIA CONVERSAZIONI PASSATE:\n{past_summaries}"

    # 8. Generazione risposta
    response = anthropic_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=enriched_prompt,
        messages=history,
    )

    risposta = response.content[0].text

    # 9. Salva risposta di Manphix nel DB
    await save_message(session_id, "assistant", risposta)

    # 10. Ogni 7 messaggi genera riassunto automatico (Livello 2)
    total_messages = await count_messages(session_id)
    if total_messages % 7 == 0:
        history.append({"role": "assistant", "content": risposta})
        await generate_and_save_summary(session_id, history, total_messages)

    return ChatResponse(response=risposta)

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    await clear_session_messages(session_id)
    return {"status": "ok"}