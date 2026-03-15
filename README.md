 # Manphix

  A web-based AI chat assistant powered by Claude and real-time web search.

  ## What it does

  Manphix is a conversational AI with a focused persona — covering Serie A football, European soccer, AI, podcasts,
  geopolitics, and American politics. It combines Claude's language capabilities with live web search via Tavily to
  provide up-to-date, sourced answers.

  ## Features

  - **AI Chat** — Powered by Claude Haiku with a custom assistant persona
  - **Web Search** — Automatically fetches current web context via Tavily API
  - **File Support** — Upload images (JPEG, PNG, GIF, WebP), PDFs, and text files
  - **Session Management** — Each user gets a unique session with persistent conversation history
  - **Password Auth** — Simple password-based login to access the chat

  ## Tech Stack

  - [FastAPI](https://fastapi.tiangolo.com/) — Backend framework
  - [Anthropic Claude](https://www.anthropic.com/) — AI model
  - [Tavily](https://tavily.com/) — Web search API
  - Deployed on [Render](https://render.com/)

  ## Setup

  1. Clone the repo:
     ```bash
     git clone https://github.com/paganid86-jpg/manphix
     cd manphix

  2. Install dependencies:
  pip install -r requirements.txt
  3. Set environment variables:
  ANTHROPIC_API_KEY=your_key
  TAVILY_API_KEY=your_key
  4. Run the app:
  uvicorn main:app --reload

  Deployment

  Configured for one-click deployment on Render via render.yaml.
