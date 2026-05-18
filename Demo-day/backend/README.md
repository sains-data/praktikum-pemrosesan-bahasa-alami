# Backend FastAPI — Demo-day Chatbot

API ringan yang memuat model **BERT** atau **Transformer** dari checkpoint di `Demo-day/models/`.

---

## Endpoints

| Method | Path | Deskripsi |
|--------|------|-----------|
| GET | `/` | Info layanan |
| GET | `/health` | Status + model ter-load |
| POST | `/chat` | Body: `{"message": "...", "beam_size": 3}` |

Contoh respons:

```json
{
  "reply": "Jalankan sudo apt update terlebih dahulu.",
  "model_type": "bert"
}
```

---

## Konfigurasi (environment)

| Variabel | Default | Arti |
|----------|---------|------|
| `MODEL_TYPE` | `bert` | `bert` atau `transformer` |
| `BERT_CHECKPOINT` | `models/bert_dream.bin` | Path relatif ke root Demo-day |
| `TRANSFORMER_CHECKPOINT` | `models/chatbot-v2.pt` | Checkpoint Transformer |
| `TRANSFORMER_VOCAB` | `models/vocab.pkl` | Kosakata char-level |

---

## Menjalankan

```bash
cd Demo-day
pip install -r requirements.txt
export MODEL_TYPE=bert
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

Dokumentasi interaktif: http://127.0.0.1:8000/docs
