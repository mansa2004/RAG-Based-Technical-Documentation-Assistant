"""
Central configuration for the RAG assistant.

All tunables live here and are overridable via environment variables (see .env.example).
Keeping this in one place makes it easy to explain/justify each default in the README.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# --- LLM provider ---
# One of: "groq", "google", "openai", "anthropic"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))

# --- Embeddings ---
# One of: "local" (sentence-transformers, no API key needed) or "openai"
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# --- Vector store ---

VECTOR_STORE_PROVIDER = os.getenv("VECTOR_STORE_PROVIDER", "chroma")


COLLECTION_NAME = os.getenv("COLLECTION_NAME", "tech_docs")
VECTOR_STORE_DIR = str(BASE_DIR / "data" / "faiss")

# --- Chunking ---
# Technical docs mix prose with code blocks, so we split on markdown structure first
# (headers, then paragraphs, then code fences) before falling back to raw character
# splitting. See README "Chunking Strategy" section for the full rationale.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

#  Retrieval 
TOP_K = int(os.getenv("TOP_K", "4"))

# Self-correction / graph control 
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))  # query rewrite + re-retrieve attempts
ENABLE_HALLUCINATION_CHECK = os.getenv("ENABLE_HALLUCINATION_CHECK", "true").lower() == "true"
ENABLE_WEB_SEARCH_FALLBACK = os.getenv("ENABLE_WEB_SEARCH_FALLBACK", "false").lower() == "true"

# Web search fallback 
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

#Feedback storage 
FEEDBACK_DB_PATH = str(BASE_DIR / "data" / "feedback.db")

# Corpus 
CORPUS_DIR = str(BASE_DIR / "corpus")
