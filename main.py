from fastapi import FastAPI, HTTPException
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
import json
from typing import List, Dict
from contextlib import asynccontextmanager

# --- CONFIGURAZIONE DATABASE ---
# DATABASE_URL va aggiunta nelle Environment Variables di Render
DATABASE_URL = os.environ.get("DATABASE_URL")

# Pool globale di connessioni al DB (creato all'avvio, riutilizzato per ogni richiesta)
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    # All'avvio: connetti al DB e crea le tabelle
    global db_pool
    db_pool = await asyncpg.create_pool(DATABASE_URL)
    await init_db()
    yield
    # Allo spegnimento: chiudi il pool
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

class LoginRequest(BaseModel):
    password: str

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    response: str

# --- FUNZIONI DB ---

async def create_session(session_id: str):
    """Salva una nuova sessione nel DB"""
    async with db_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO sessions (session_id) VALUES ($1)", session_id
        )

async def session_exists(session_id: str) -> bool:
    """Controlla se la sessione esiste nel DB"""
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT session_id FROM sessions WHERE session_id = $1", session_id
        )
        return row is not None

async def save_message(session_id: str, role: str, content: str):
    """Salva un messaggio nel DB"""
    async with db_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES ($1, $2, $3)",
            session_id, role, content
        )

async def get_history(session_id: str) -> List[Dict]:
    """Recupera la storia della conversazione dal DB"""
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT role, content FROM messages WHERE session_id = $1 ORDER BY created_at ASC",
            session_id
        )
        return [{"role": r["role"], "content": r["content"]} for r in rows]

async def clear_session_messages(session_id: str):
    """Cancella i messaggi di una sessione"""
    async with db_pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM messages WHERE session_id = $1", session_id
        )

# --- FUNZIONI DI RECUPERO CONOSCENZA ---

async def get_github_file_content(file_path: str):
    """Scarica il contenuto raw di un singolo file"""
    url = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/main/{file_path}"
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(url, headers=HEADERS)
            return resp.text if resp.status_code == 200 else ""
        except:
            return ""

async def get_all_learnings():
    """Scarica e concatena tutti i file nella cartella .learnings"""
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
async def chat(req: ChatRequest):
    if not await session_exists(req.session_id):
        raise HTTPException(status_code=401, detail="Sessione non valida")

    # 1. Recupero dinamico della conoscenza da GitHub
    knowledge_main = await get_github_file_content("CLAUDE.md")
    knowledge_folder = await get_all_learnings()
    full_external_knowledge = f"{knowledge_main}\n{knowledge_folder}"

    # 2. Recupero storia dal DB (invece che dalla RAM)
    history = await get_history(req.session_id)

    # 3. Ricerca Web (Tavily)
    try:
        risultati = tavily_client.search(
            req.message,
            max_results=5,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=False
        )
        contesto = "".join([f"- {r['title']}: {r['content']}\n\n" for r in risultati["results"]])
    except:
        contesto = ""

    messaggio = req.message
    if contesto and len(contesto.strip()) > 50:
        messaggio = f"Contesto Web (usa solo se pertinente alla domanda):\n{contesto}\n\nDomanda: {req.message}"

    # 4. Salva messaggio utente nel DB
    await save_message(req.session_id, "user", messaggio)
    history.append({"role": "user", "content": messaggio})

    # 5. Generazione risposta
    enriched_prompt = f"{SYSTEM_PROMPT}\n\n### KNOWLEDGE BASE ESTERNA (I tuoi apprendimenti):\n{full_external_knowledge}"

    response = anthropic_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=enriched_prompt,
        messages=history,
    )

    risposta = response.content[0].text

    # 6. Salva risposta di Manphix nel DB
    await save_message(req.session_id, "assistant", risposta)

    return ChatResponse(response=risposta)

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    await clear_session_messages(session_id)
    return {"status": "ok"}