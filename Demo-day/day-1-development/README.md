# Hari 1 — Development

**Demo-day · Sesi 1 dari 2**  
**Topik:** Chatbot helpdesk teknis · **Dataset:** [Ubuntu Dialogue Corpus](https://github.com/rkadlec/ubuntu-ranking-dataset-creator)

Pada hari ini Anda menyiapkan data, melatih model (**BERT** atau **Transformer**), dan menyerahkan checkpoint ke `Demo-day/models/` untuk dipakai besok di deployment.

> **Hari berikutnya:** [`../day-2-deployment/README.md`](../day-2-deployment/README.md) — FastAPI + Discord.

---

## 1. Siapkan data

### Opsi A — Sampel cepat (demo kelas)

```bash
cd Demo-day/day-1-development
python3 scripts/prepare_sample_data.py
```

Keluaran di `data/`:

- `chatbot_bert.txt` — format `pertanyaan=jawaban` (BERT)  
- `dialog_transformer.txt` — format `pertanyaan|jawaban` (Transformer)

Salin ke modul latih:

```bash
cp data/chatbot_bert.txt ../../../Chatbot/Bert_chatbot/data/chatbot.txt
```

### Opsi B — Ubuntu Dialogue Corpus (laporan/tugas)

1. Clone [ubuntu-ranking-dataset-creator](https://github.com/rkadlec/ubuntu-ranking-dataset-creator).
2. Bangun pasangan Q–A satu turn dari dialog training.
3. Konversi:

   ```bash
   python3 scripts/convert_ubuntu_to_chatbot.py \
     --input /path/to/udc_pairs.tsv \
     --output_dir ./data
   ```

4. Untuk bahasa Indonesia: terjemahkan subset 5k–20k pasangan, simpan sebagai `chatbot_bert.txt`.

Detail format: [`data/README.md`](data/README.md).

---

## 2. Jalur A — BERT (opsi 1)

Panduan modul: [`Chatbot/Bert_chatbot/README.md`](../../Chatbot/Bert_chatbot/README.md)

```bash
cd ../../../Chatbot/Bert_chatbot

# Wajib ada: data/chatbot.txt, data/vocab.txt, data/pytorch_model.bin
pip install torch transformers tqdm

python3 train.py
```

Setelah latih, salin bobot:

```bash
mkdir -p ../../../Demo-day/models
cp bert_dream.bin ../../../Demo-day/models/bert_dream.bin
```

**Demo live di kelas — uji konsol:**

```bash
python3 interface.py
```

Contoh input: `WiFi kampus tidak bisa connect?`

---

## 3. Jalur B — Transformer (opsi 2)

Panduan: [`Chatbot/transformer_chatbot/README.md`](../../Chatbot/transformer_chatbot/README.md)

```bash
cd ../../../Chatbot/transformer_chatbot

# Sesuaikan train_filename di config.py
python3 pre_process.py
python3 train.py
python3 export.py
```

Salin artefak:

```bash
mkdir -p ../../../Demo-day/models
cp chatbot-v2.pt ../../../Demo-day/models/
cp data/vocab.pkl ../../../Demo-day/models/
```

Uji: `python3 chat.py`

---

## 4. Metrik & evaluasi (untuk slide)

| Metrik | Cara |
|--------|------|
| Loss latih/valid | Log dari `train.py` |
| Contoh dialog | 5–10 pertanyaan uji manual di `interface.py` / `chat.py` |
| BLEU/ROUGE (opsional) | Bandingkan jawaban model vs referensi di data uji |

---

## Checklist penutup Hari 1

Centang sebelum menutup sesi — ini yang dibawa ke **Hari 2**:

- [ ] Dataset open source disebutkan (UDC + URL di slide)  
- [ ] `Demo-day/models/bert_dream.bin` (atau `chatbot-v2.pt` + `vocab.pkl`) ada  
- [ ] `Chatbot/Bert_chatbot/data/vocab.txt` dan `pytorch_model.bin` tidak hilang (dibutuhkan API)  
- [ ] Minimal 3 contoh Q–A yang jawabannya masuk akal (screenshot untuk laporan)  
- [ ] Tim tahu `MODEL_TYPE=bert` atau `transformer` yang dipakai besok 

---

## Troubleshooting

| Masalah | Solusi |
|---------|--------|
| `pytorch_model.bin` tidak ada | Unduh BERT-Chinese; letakkan di `Bert_chatbot/data/` |
| OOM saat latih | Kurangi `batch_size` di `config.py` |
| Loss tidak turun | Periksa format `chatbot.txt` (`=` pemisah) |
| Transformer gagal | Jalankan ulang `pre_process.py`, cek `vocab.pkl` |

---

## Dependensi Hari 1

```bash
cd Demo-day
pip install -r requirements-day1.txt
```

Atau paket lengkap: `pip install -r requirements.txt` dari root `Demo-day/`.
