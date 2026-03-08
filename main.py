from fastapi import FastAPI, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic
from tavily import TavilyClient
import os
import httpx
from typing import List, Dict

app = FastAPI()

# --- CONFIGURAZIONE GITHUB ---
# Aggiungi GITHUB_TOKEN nelle Environment Variables di Render per evitare limiti di velocità
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

# Sessioni in memoria
sessions: Dict[str, List[Dict]] = {}

class LoginRequest(BaseModel):
    password: str

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    response: str

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
    import uuid
    session_id = str(uuid.uuid4())
    sessions[session_id] = []
    return {"session_id": session_id}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if req.session_id not in sessions:
        raise HTTPException(status_code=401, detail="Sessione non valida")

    # 1. Recupero dinamico della conoscenza da GitHub
    knowledge_main = await get_github_file_content("CLAUDE.md")
    knowledge_folder = await get_all_learnings()
    full_external_knowledge = f"{knowledge_main}\n{knowledge_folder}"

    history = sessions[req.session_id]

    # 2. Ricerca Web (Tavily) — FIX: advanced + più risultati
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

    # FIX: passa il contesto solo se è sostanzioso, e dì a Manphix di usarlo solo se pertinente
    messaggio = req.message
    if contesto and len(contesto.strip()) > 50:
        messaggio = f"Contesto Web (usa solo se pertinente alla domanda):\n{contesto}\n\nDomanda: {req.message}"

    history.append({"role": "user", "content": messaggio})

    # 3. Generazione risposta con Prompt arricchito
    enriched_prompt = f"{SYSTEM_PROMPT}\n\n### KNOWLEDGE BASE ESTERNA (I tuoi apprendimenti):\n{full_external_knowledge}"

    response = anthropic_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=enriched_prompt,
        messages=history,
    )

    risposta = response.content[0].text
    history.append({"role": "assistant", "content": risposta})
    sessions[req.session_id] = history

    return ChatResponse(response=risposta)

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    if session_id in sessions:
        sessions[session_id] = []
    return {"status": "ok"}