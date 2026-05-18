# Hands-On: Chatbot Transformer (Encoder–Decoder)

**Mata kuliah:** Pemrosesan Bahasa Alami Berbasis Transformer  
**Program Studi Sains Data · Fakultas Sains · Institut Teknologi Sumatera**

Modul ini mengimplementasikan chatbot dengan arsitektur **Transformer** standar (Vaswani et al.): encoder stack + decoder stack, multi-head self-attention, positional encoding, dan pelatihan pada korpus dialog skala besar (contoh asli: Qingyun ~120 ribu pasangan, Bahasa Mandarin).

---

## 1. Teori dan konsep

### 1.1 Mengapa Transformer untuk dialog?

Berbeda dengan RNN, Transformer memproses seluruh urutan secara paralel dan memodelkan ketergantungan jarak jauh lewat **self-attention**. Untuk chatbot:

- **Encoder** memahami pertanyaan pengguna.
- **Decoder** menghasilkan jawaban token demi token dengan **masked self-attention** (tidak melihat masa depan) dan **cross-attention** ke keluaran encoder.

### 1.2 Arsitektur

```
  Input (karakter/token id)          Target (jawaban, shift)
           │                                    │
           ▼                                    ▼
  ┌─────────────────┐              ┌─────────────────┐
  │ Encoder × N     │              │ Decoder × N     │
  │ · Self-Attn     │──context────►│ · Masked Self   │
  │ · FFN           │              │ · Cross-Attn    │
  │ · Pos Encoding  │              │ · FFN           │
  └─────────────────┘              └────────┬────────┘
                                            ▼
                                   Linear → vocab (softmax)
```

**Struktur folder:**

| Path | Peran |
|------|--------|
| `transformer/encoder.py` | Encoder stack |
| `transformer/decoder.py` | Decoder stack |
| `transformer/attention.py` | Scaled dot-product, multi-head |
| `transformer/transformer.py` | Gabungan + `recognize()` untuk decoding |
| `transformer/loss.py` | Cross-entropy + label smoothing |
| `transformer/optimizer.py` | LR schedule Noam (warmup) |
| `pre_process.py` | Bangun kosakata karakter & simpan `.pkl` |
| `data_gen.py` | Dataset PyTorch + `pad_collate` |
| `train.py` | Latih + validasi + uji sampel |
| `export.py` | Ekspor `BEST_checkpoint.tar` → `chatbot-v2.pt` |
| `chat.py` | Chat interaktif |

### 1.3 Representasi karakter

Model memakai **kosakata tingkat karakter** (`char2idx`), cocok untuk bahasa Mandarin tanpa tokenisasi WordPiece. Untuk Indonesia bisa tetap dipakai (per huruf) atau dimodifikasi ke subkata.

### 1.4 Optimizer

`TransformerOptimizer` menerapkan skema learning rate dari paper asli: linear warmup lalu decay proporsional terhadap `step^{-0.5}`.

---

## 2. Data dan prasyarat

### 2.1 Format mentah

File CSV/teks: `pertanyaan | jawaban` per baris (pemisah `|`):

```text
Halo | Hai, ada yang bisa dibantu?
Cuaca hari ini bagaimana? | Cerah dan cerah.
```

Default di `config.py`:

```python
train_filename = './data/12万对话语料青云库.csv'
```

Ganti path ke dataset Anda jika tidak memakai korpus Qingyun.

### 2.2 Artefak setelah prapemrosesan

| Berkas | Isi |
|--------|-----|
| `data/vocab.pkl` | `char2idx`, `idx2char` |
| `data/data.pkl` | `train`, `valid`, `test` (list sampel) |
| `data/lengths.npy` | Panjang tiap cuplikan (analisis) |

### 2.3 Dependensi

```bash
pip install torch numpy tqdm tensorboard
```

---

## 3. Langkah hands-on

### Langkah 1 — Prapemrosesan

```bash
cd Chatbot/transformer_chatbot
python pre_process.py
```

Periksa log: ukuran kosakata, jumlah train/valid/test.

### Langkah 2 — Pelatihan

```bash
python train.py --batch-size 2 --epochs 50
```

Opsi penting (lihat `utils.parse_args`):

| Argumen | Default | Keterangan |
|---------|---------|------------|
| `--n_layers_enc` | 6 | Lapisan encoder |
| `--n_layers_dec` | 6 | Lapisan decoder |
| `--n_head` | 8 | Kepala attention |
| `--d_model` | 512 | Dimensi model |
| `--label_smoothing` | 0.1 | Regularisasi label |
| `--warmup_steps` | 4000 | Warmup optimizer |

**Keluaran pelatihan:**

- `checkpoint.tar` — checkpoint terakhir  
- `BEST_checkpoint.tar` — validasi loss terbaik  
- Log TensorBoard: `tensorboard --logdir runs/`

### Langkah 3 — Validasi dan uji otomatis

Setiap epoch, `train.py` memanggil `valid()` (loss validasi) dan `test()` (beberapa sampel dari set uji, dicetak ke log).

### Langkah 4 — Ekspor bobot untuk inferensi

```bash
python export.py
```

Menghasilkan `chatbot-v2.pt` (hanya `state_dict`) dari `BEST_checkpoint.tar`.

### Langkah 5 — Chat interaktif

```bash
python chat.py
```

Ketik `q` untuk keluar.

---

## 4. Metrik evaluasi

### 4.1 Metrik bawaan

| Metrik | Lokasi | Arti |
|--------|--------|------|
| **Train loss** | `train()` | CE (+ label smoothing) pada token target |
| **Valid loss** | `valid()` | Sama, tanpa update bobot; dipakai pilih model terbaik |
| **Token accuracy** | `cal_performance()` | `n_correct` / token non-PAD (bisa ditambahkan ke log) |

Rumus loss dengan smoothing ada di `transformer/loss.py`.

### 4.2 Metrik tambahan (disarankan laporan)

1. **Perplexity validasi** — `exp(valid_loss)`.
2. **BLEU / chrF** — pada subset `test` setelah decoding `model.recognize()`.
3. **Diversity** — distinct-1 / distinct-2 pada jawaban model (cegah jawaban generik).

Contoh akurasi token dari return `cal_performance`:

```python
pred, gold = model(...)
loss, n_correct = cal_performance(pred, gold, smoothing=0.1)
# tambahkan hitung total token non-PAD untuk rasio akurasi
```

### 4.3 Checklist laporan

- [ ] Kurva train/valid loss (TensorBoard)
- [ ] Tabel 10 sampel: input | referensi | prediksi model
- [ ] Pengaruh `label_smoothing` (0 vs 0.1)
- [ ] Perbandingan dengan `seq2seq_luong` pada data subsample yang sama

---

## 5. Deploy model

### 5.1 Alur deploy standar modul ini

```
pre_process.py → train.py → BEST_checkpoint.tar
       → export.py → chatbot-v2.pt → chat.py
```

### 5.2 Inferensi programatik

```python
import torch
import pickle
from transformer.transformer import Transformer
from config import Config

model = Transformer()
model.load_state_dict(torch.load('chatbot-v2.pt', map_location='cpu'))
model.eval()

with open(Config.vocab_file, 'rb') as f:
    idx2char = pickle.load(f)['dict']['idx2char']
    char2idx = pickle.load(f)['dict']['char2idx']

def jawab(teks):
    ids = [char2idx.get(c, Config.unk_id) for c in teks]
    inp = torch.tensor(ids, dtype=torch.long)
    length = torch.tensor([len(ids)])
    hyps = model.recognize(input=inp, input_length=length, char_list=idx2char)
    out = ''.join(idx2char[i] for i in hyps[0]['yseq'])
    return out.replace('<sos>', '').replace('<eos>', '')
```

### 5.3 API FastAPI (contoh)

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
# muat model sekali saat startup ...

class ChatRequest(BaseModel):
    text: str

@app.post("/chat")
def chat(req: ChatRequest):
    return {"reply": jawab(req.text)}
```

```bash
pip install fastapi uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 5.4 Produksi

| Aspek | Rekomendasi |
|-------|-------------|
| GPU | Batch=1 inference; FP16 jika didukung |
| Batas panjang | `maxlen_in`, `maxlen_out` di `Config` |
| Monitoring | Latensi p95, OOV rate (`unk_id`) |
| Versi | Simpan `vocab.pkl` + `chatbot-v2.pt` berpasangan |

---

## 6. Konfigurasi (`config.py`)

| Parameter | Default | Keterangan |
|-----------|---------|------------|
| `maxlen_in` / `maxlen_out` | 50 | Panjang maksimum |
| `vocab_size` | 5884 | Diperbarui setelah `pre_process.py` |
| `grad_clip` | 1.0 | Clipping gradien |
| `print_freq` | 50 | Frekuensi log batch |
| `IGNORE_ID` | -1 | Token diabaikan dalam loss |

---

## 7. Troubleshooting

| Masalah | Solusi |
|---------|--------|
| `FileNotFoundError` data.pkl | Jalankan `pre_process.py` dulu |
| `chat.py` NameError | Pastikan variabel `sentence_in` konsisten (perbaikan di versi terbaru) |
| Loss NaN | Kurangi LR, periksa `grad_clip`, kurangi `batch-size` |
| Jawaban karakter acak | Epoch belum cukup; periksa alignment format `|` pada data |
| OOM | Kurangi `batch-size`, `d_model`, atau jumlah layer |

---

## Referensi

- Vaswani et al., *Attention Is All You Need* (2017)  
- Korpus Qingyun (dialog Mandarin) — disebut di README asli  
- Kode asli: [shawroad/NLP_pytorch_project](https://github.com/shawroad/NLP_pytorch_project) — `Chatbot/transformer_chatbot`
