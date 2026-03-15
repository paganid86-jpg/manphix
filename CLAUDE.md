# CLAUDE.md — Manphix Project

## Panoramica

Manphix è un assistente AI personale con estetica cyberpunk/sci-fi. È composto da un backend FastAPI e un frontend single-file HTML. Il progetto è di Dante Pagani (@paganid86-jpg su GitHub).

## Tono e stile di lavoro

- **Spiega le scelte importanti** (perché usiamo un certo approccio, trade-off architetturali), ma resta rapido sul resto.
- Quando modifichi codice esistente, indica brevemente cosa cambia e perché.
- Lingua preferita per comunicare: **italiano**.
- Produci file pronti da usare, non snippet parziali.

---

## Architettura Backend

- **Framework**: FastAPI (`app.py`)
- **LLM**: Claude Haiku (Anthropic API)
- **Web Search**: Tavily API (advanced depth, query preprocessing)
- **Database**: PostgreSQL su Render
  - Tabelle principali: `sessions`, `messages`, `summaries`
  - Implementa un sistema di memoria a 3 livelli cross-session
- **Repo GitHub**: `paganid86-jpg/manphix`

### Variabili d'ambiente (Render)

| Variabile | Uso |
|---|---|
| `ANTHROPIC_API_KEY` | API Claude |
| `TAVILY_API_KEY` | Ricerca web Tavily |
| `GITHUB_TOKEN` | Accesso repo GitHub |
| `MANPHIX_PASSWORD` | Autenticazione login |
| `DATABASE_URL` | Connessione PostgreSQL |

> ⚠️ Non hardcodare mai chiavi o segreti nel codice. Usa sempre variabili d'ambiente.

### Knowledge Base

- Repo companion: `paganid86-jpg/manphix-brain`
- File knowledge base: `.learnings/dante.md`

---

## Architettura Frontend

- **File principale**: `static/index.html` (single-file)
- **Tema**: dark, palette purple/indigo cyberpunk
- **Componenti UI**:
  - Canvas aurora animato come sfondo
  - Schermata di login
  - Interfaccia chat
  - Neural sphere animation come indicatore di "thinking"
- **Libreria UI suggerita**: Aceternity UI (per l'estetica)
- **Feature attive/in sviluppo**: streaming risposte, sidebar cronologia chat, syntax highlighting

### Regole frontend

- Tutto in un singolo file HTML (CSS e JS inline) — no file separati se non strettamente necessario.
- Mantieni le animazioni performanti (usa `requestAnimationFrame`, evita layout thrashing).
- Rispetta la palette colori esistente: viola, indigo, toni scuri.

---

## Workflow di Deploy

1. **Sviluppo**: modifica locale → test → commit su GitHub
2. **Deploy**: Render auto-deploya dal branch principale del repo `manphix`
3. **Approccio iterativo**: una modifica alla volta → deploy → verifica che funzioni → prossima modifica
4. Non accumulare troppe modifiche in un singolo deploy.

### Comandi utili

```bash
# Avvio locale del backend
uvicorn app:app --reload --port 8000

# Push su GitHub (triggera deploy su Render)
git add .
git commit -m "descrizione chiara della modifica"
git push origin main
```

---

## Regole generali

- **Non rompere ciò che funziona**: prima di modificare una feature esistente, verifica il comportamento attuale.
- **Commenti nel codice**: brevi e utili, in italiano o inglese (consistente con il file).
- **Error handling**: gestisci sempre gli errori delle API esterne (Anthropic, Tavily) con try/except e messaggi chiari.
- **Sicurezza**: mai esporre chiavi API al frontend, mai loggare dati sensibili.
