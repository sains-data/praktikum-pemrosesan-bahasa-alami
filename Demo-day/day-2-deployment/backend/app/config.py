"""Konfigurasi backend Demo-day."""
import os
from pathlib import Path

# Demo-day/ (root paket): app → backend → day-2-deployment → Demo-day
DEMO_DAY_ROOT = Path(__file__).resolve().parents[3]
CHATBOT_ROOT = DEMO_DAY_ROOT.parent / "Chatbot"

MODEL_TYPE = os.getenv("MODEL_TYPE", "bert").lower().strip()
BERT_CHECKPOINT = DEMO_DAY_ROOT / os.getenv("BERT_CHECKPOINT", "models/bert_dream.bin")
TRANSFORMER_CHECKPOINT = DEMO_DAY_ROOT / os.getenv("TRANSFORMER_CHECKPOINT", "models/chatbot-v2.pt")
TRANSFORMER_VOCAB = DEMO_DAY_ROOT / os.getenv("TRANSFORMER_VOCAB", "models/vocab.pkl")

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
