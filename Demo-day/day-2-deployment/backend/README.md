# Backend FastAPI — Hari 2

API inferensi yang memuat checkpoint dari `Demo-day/models/`.

---

## Endpoints

| Method | Path | Deskripsi |
|--------|------|-----------|
| GET | `/` | Info layanan |
| GET | `/health` | Status + model ter-load |
| POST | `/chat` | `{"message": "...", "beam_size": 3}` |

---

## Menjalankan

```bash
cd Demo-day/day-2-deployment
export MODEL_TYPE=bert
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

Dokumentasi: http://127.0.0.1:8000/docs

Panduan lengkap + Discord: [`../README.md`](../README.md).
