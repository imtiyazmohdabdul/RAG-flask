<<<<<<< HEAD
# RAG-flask
=======
# Antigravity RAG (Local)

A premium, fully offline RAG system using Flask, ChromaDB, and DeepSeek-R1.

## ⚡️ Quick Start

### 1. Requirements
Ensure you have **Python 3.10+** and **Ollama** installed.

```bash
# Install Python deps
pip install -r requirements.txt

# Pull the model (Required once)
ollama pull deepseek-r1:14b
ollama pull all-minilm
```

### 2. Run
1. Start Ollama in a separate terminal:
   ```bash
   ollama serve
   ```
2. Start the App:
   ```bash
   python app.py
   ```
3. Open `http://localhost:5000` in your browser.

## 🧪 Features
- **Real-time Ingestion**: Drag & drop PDF/TXT files and watch them index instantly.
- **Streaming Chat**: Responses flow in token-by-token.
- **Scientific Citations**: The model cites the exact filename it retrieved info from.
- **No Cloud**: 100% local execution.
>>>>>>> b3d4239 (anything)
