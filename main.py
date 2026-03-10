from fastapi import FastAPI, HTTPException, Form, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic
from tavily import TavilyClient
import os
import base64
from typing import List, Dict, Optional

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

SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}

@app.post("/chat", response_model=ChatResponse)
async def chat(
    session_id: str = Form(...),
    message: str = Form(default=""),
    file: Optional[UploadFile] = File(default=None),
):
    if session_id not in sessions:
        raise HTTPException(status_code=401, detail="Sessione non valida")

    history = sessions[session_id]

    # Web search solo se c'è un messaggio testuale
    contesto = ""
    if message.strip():
        try:
            risultati = tavily_client.search(message, max_results=3, search_depth="basic")
            for r in risultati["results"]:
                contesto += f"- {r['title']}: {r['content']}\n\n"
        except:
            contesto = ""

    text_message = message
    if contesto:
        text_message = f"Contesto aggiornato dal web:\n{contesto}\n\nDomanda: {message}"

    # Costruisce i content block
    content_blocks: List[dict] = []

    if file and file.filename:
        file_bytes = await file.read()
        mime = file.content_type or ""

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
            text_message = f"[File allegato: {file.filename}]\n```\n{file_text}\n```\n\n{text_message}".strip()

    content_blocks.append({"type": "text", "text": text_message or "Analizza questo contenuto."})
    history.append({"role": "user", "content": content_blocks})

    response = anthropic_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=history,
    )

    risposta = response.content[0].text
    history.append({"role": "assistant", "content": risposta})
    sessions[session_id] = history

    return ChatResponse(response=risposta)

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    if session_id in sessions:
        sessions[session_id] = []
    return {"status": "ok"}
