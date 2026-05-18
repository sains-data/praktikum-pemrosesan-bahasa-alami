# Demo-day — Panduan Development

**Topik:** Chatbot helpdesk teknis  
**Dataset:** [Ubuntu Dialogue Corpus](https://github.com/rkadlec/ubuntu-ranking-dataset-creator) (+ sampel Indonesia di `data/`)

---

## 1. Siapkan data

### Opsi A — Sampel cepat (tanpa unduh UDC penuh)

Gunakan file contoh untuk uji pipeline:

```bash
cd Demo-day/development
python scripts/prepare_sample_data.py
```

Keluaran:

- `data/chatbot_bert.txt` — format `pertanyaan=jawaban` (untuk BERT)  
- `data/dialog_transformer.txt` — format `pertanyaan|jawaban` (untuk Transformer)

Salin ke folder latih:

```bash
cp data/chatbot_bert.txt ../../Chatbot/Bert_chatbot/data/chatbot.txt
```

### Opsi B — Ubuntu Dialogue Corpus (disarankan laporan)

1. Clone repo UDC dan ikuti instruksi pembuatannya:

   ```bash
   git clone https://github.com/rkadlec/ubuntu-ranking-dataset-creator.git
   ```

2. Bangun pasangan Q–A (satu turn) dari folder `dialogueTraining` / test sesuai dokumentasi repo.

3. Konversi ke format proyek:

   ```bash
   python scripts/convert_ubuntu_to_chatbot.py \
     --input /path/to/udc_pairs.tsv \
     --output_dir ./data
   ```

4. **Bahasa Indonesia:** terjemahkan subset 5k–20k pasangan (Google Translate API, Marian, atau manual untuk demo), lalu simpan sebagai `chatbot_bert.txt`.

---

## 2. Jalur A — BERT (disarankan untuk Discord)

Panduan lengkap: [`Chatbot/Bert_chatbot/README.md`](../../Chatbot/Bert_chatbot/README.md)

```bash
cd ../../Chatbot/Bert_chatbot

# Pastikan ada: data/chatbot.txt, data/vocab.txt, data/pytorch_model.bin (BERT Chinese/base)
pip install torch transformers tqdm

python train.py
# Setelah selesai, checkpoint biasanya bert_dream.bin atau sesuai train.py
```

Salin bobot ke Demo-day:

```bash
mkdir -p ../../Demo-day/models
cp bert_dream.bin ../../Demo-day/models/bert_dream.bin
```

Uji lokal:

```bash
python interface.py
```

---

## 3. Jalur B — Transformer

Panduan: [`Chatbot/transformer_chatbot/README.md`](../../Chatbot/transformer_chatbot/README.md)

```bash
cd ../../Chatbot/transformer_chatbot

# 1. Sesuaikan train_filename di config.py ke CSV/dialog Anda (format pertanyaan|jawaban)
python pre_process.py
python train.py
python export.py   # hasil: chatbot-v2.pt
```

Salin artefak:

```bash
mkdir -p ../../Demo-day/models
cp chatbot-v2.pt ../../Demo-day/models/
cp data/vocab.pkl ../../Demo-day/models/
cp data/data.pkl ../../Demo-day/models/   # jika diperlukan engine
```

Uji:

```bash
python chat.py
```

---

## 4. Uji API sebelum Discord

```bash
cd ../../Demo-day
export MODEL_TYPE=bert   # atau transformer
uvicorn backend.app.main:app --reload --port 8000
```

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Bagaimana cara update sistem?"}'
```

---

## 5. Metrik untuk laporan demo

| Metrik | BERT | Transformer |
|--------|------|-------------|
| Loss validasi | CE pada token jawaban | CE + label smoothing |
| BLEU / ROUGE (opsional) | `nltk` vs referensi | Sama |
| Latensi P95 | Ukur dari `/chat` | Sama |
| Contoh dialog | 10 pasangan uji manual | Sama |

**Checklist demo development:**

- [ ] Dataset open source disebutkan (UDC + URL)  
- [ ] Minimal 1.000 pasangan latih (subset)  
- [ ] Checkpoint tersimpan di `Demo-day/models/`  
- [ ] POST `/chat` mengembalikan jawaban masuk akal  
- [ ] Screenshot percakapan Discord  

---

## 6. Troubleshooting

| Masalah | Solusi |
|---------|--------|
| `pytorch_model.bin` tidak ada | Unduh BERT-base / mengzi dari HuggingFace, konversi ke format proyek |
| OOM saat latih BERT | Kurangi `batch_size` di `config.py` |
| Transformer loss tidak turun | Periksa `pre_process.py`, kurangi `max_seq_length` |
| API `Model not loaded` | Cek `MODEL_TYPE` dan path file di `models/` |
