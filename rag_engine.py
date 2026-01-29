import os
import shutil
import uuid
import logging
from pathlib import Path
from typing import List, Dict, Generator
import requests
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from werkzeug.utils import secure_filename

# --- Configuration ---
PERSIST_DIRECTORY = os.path.join(os.getcwd(), "data", "db")
UPLOAD_FOLDER = os.path.join(os.getcwd(), "data", "uploads")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# API / LLM Config
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if DEEPSEEK_API_KEY:
    LLM_API_URL = "https://api.deepseek.com/chat/completions"
    LLM_MODEL = "deepseek-chat" # or deepseek-reasoner
    logger.info("Using DeepSeek API")
else:
    LLM_API_URL = "http://localhost:11434/api/generate"
    LLM_MODEL = "deepseek-r1:14b"
    logger.info("Using Local Ollama")

# --- RAG Engine Class ---
class RAGEngine:
    def __init__(self):
        self._ensure_directories()
        
        logger.info("Initializing Embedding Model...")
        self.embedding_fn = SentenceTransformer(EMBEDDING_MODEL)
        
        logger.info(f"Initializing Vector DB at {PERSIST_DIRECTORY}...")
        self.chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        self.collection = self.chroma_client.get_or_create_collection(name="scientific_knowledge")

    def _ensure_directories(self):
        os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    def ingest_file(self, file_path: str, filename: str) -> int:
        """Processes a file: converts to text, chunks, embeds, and stores."""
        text = self._extract_text(file_path)
        if not text:
            raise ValueError("No text extracted from file.")

        chunks = self._chunk_text(text)
        if not chunks:
            return 0
        
        # Prepare data for Chroma
        ids = [str(uuid.uuid4()) for _ in chunks]
        embeddings = self.embedding_fn.encode(chunks).tolist()
        metadatas = [{"source": filename, "chunk_id": i} for i in range(len(chunks))]
        
        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )
        return len(chunks)

    def _extract_text(self, file_path: str) -> str:
        ext = Path(file_path).suffix.lower()
        text = ""
        try:
            if ext == '.pdf':
                reader = PdfReader(file_path)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            elif ext in ['.txt', '.md']:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
        return text

    def _chunk_text(self, text: str) -> List[str]:
        chunks = []
        for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk = text[i:i + CHUNK_SIZE]
            if len(chunk) > 100: # Filter tiny chunks
                chunks.append(chunk)
        return chunks

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieves top-k relevant chunks."""
        query_emb = self.embedding_fn.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_emb,
            n_results=k
        )
        
        hits = []
        if results['documents']:
            for i in range(len(results['documents'][0])):
                hits.append({
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i] if results['distances'] else 0
                })
        return hits

    def generate_stream(self, query: str, context_chunks: List[Dict]) -> Generator[str, None, None]:
        """Generates a streaming response from LLM (DeepSeek API or Local Ollama)."""
        
        # Format context
        context_str = "\n\n".join([f"[Source: {c['metadata']['source']}] {c['content']}" for c in context_chunks])
        
        system_prompt = """Directory: Antigravity Research
Role: Scientific Assistant
Objective: Answer based ONLY on the provided Context.
Instructions:
1. Valid references: Cite specific filenames from context.
2. If uncertain, state "Insufficient data in current knowledge base."
3. Think step-by-step."""

        user_content = f"""Context:
{context_str}

User Question: {query}
"""

        # DeepSeek API (OpenAI Compatible)
        if DEEPSEEK_API_KEY:
            headers = {
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                "stream": True,
                "temperature": 0.2
            }
            
            try:
                with requests.post(LLM_API_URL, json=payload, headers=headers, stream=True) as r:
                    r.raise_for_status()
                    for line in r.iter_lines():
                        if line:
                            decoded_line = line.decode('utf-8').strip()
                            if decoded_line.startswith("data: "):
                                data_str = decoded_line[6:]
                                if data_str == "[DONE]":
                                    break
                                try:
                                    import json
                                    json_resp = json.loads(data_str)
                                    delta = json_resp.get("choices", [{}])[0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        yield content
                                except:
                                    pass
            except Exception as e:
                yield f"\n[API Error: {str(e)}]"

        # Local Ollama Fallback
        else:
            prompt = f"{system_prompt}\n\n{user_content}\n\nAnswer:\n"
            
            payload = {
                "model": LLM_MODEL,
                "prompt": prompt,
                "stream": True,
                "options": {"temperature": 0.2, "num_ctx": 4096}
            }

            try:
                with requests.post(LLM_API_URL, json=payload, stream=True) as r:
                    r.raise_for_status()
                    for line in r.iter_lines():
                        if line:
                            body = line.decode('utf-8')
                            if body.strip():
                                import json
                                try:
                                    json_resp = json.loads(body)
                                    word = json_resp.get("response", "")
                                    if word:
                                        yield word
                                except:
                                    pass
            except requests.exceptions.RequestException as e:
                yield f"\n[System Error: Could not connect to Ollama. Ensure 'ollama run {LLM_MODEL}' is active or set DEEPSEEK_API_KEY for cloud hosting.]"

# Singleton instance
rag = RAGEngine()
