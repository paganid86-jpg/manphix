from fastapi import FastAPI, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic
from tavily import TavilyClient
import os
from typing import List, Dict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Password di accesso
ACCESS_PASSWORD = os.environ.get("MANPHIX_PASSWORD", "manphix2024")

# API clients
anthropic_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
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

# Sessioni in memoria
sessions: Dict[str, List[Dict]] = {}

class LoginRequest(BaseModel):
    password: str

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    response: str

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

    history = sessions[req.session_id]

    # Cerca sul web
    try:
        risultati = tavily_client.search(req.message, max_results=3, search_depth="basic")
        contesto = ""
        for r in risultati["results"]:
            contesto += f"- {r['title']}: {r['content']}\n\n"
    except:
        contesto = ""

    messaggio = req.message
    if contesto:
        messaggio = f"Contesto aggiornato dal web:\n{contesto}\n\nDomanda: {req.message}"

    history.append({"role": "user", "content": messaggio})

    response = anthropic_client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
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
