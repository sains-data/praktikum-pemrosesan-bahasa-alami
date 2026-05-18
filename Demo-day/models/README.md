# Folder checkpoint Demo-day

Letakkan file model hasil latih di sini agar backend FastAPI dapat memuatnya.

| File | Jalur latih | Cara mendapatkan |
|------|-------------|------------------|
| `bert_dream.bin` | `Chatbot/Bert_chatbot/` | Setelah `train.py`, salin `bert_dream.bin` ke folder ini |
| `chatbot-v2.pt` | `Chatbot/transformer_chatbot/` | Setelah `export.py` |
| `vocab.pkl` | `transformer_chatbot/data/` | Setelah `pre_process.py` |

**BERT** juga membutuhkan file di `Chatbot/Bert_chatbot/data/` (bukan di folder ini):

- `vocab.txt`
- `pytorch_model.bin` (BERT pra-latih)

Lihat [`day-1-development/README.md`](../day-1-development/README.md) (latih) dan [`day-2-deployment/README.md`](../day-2-deployment/README.md) (deploy).
