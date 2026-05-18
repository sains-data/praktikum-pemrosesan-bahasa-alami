# Hands-On: Distilasi BERT → BiLSTM (Soft Label)

**Mata kuliah:** Pemrosesan Bahasa Alami Berbasis Transformer  
**Program Studi Sains Data · Fakultas Sains · Institut Teknologi Sumatera**

Modul paling intuitif untuk distilasi: **guru** = BERT klasifikasi teks, **murid** = BiLSTM ringan dengan embedding Keras. Murid belajar dari **label keras** (0/1) dan **label lunak** (distribusi softmax guru).

---

## 1. Teori dan konsep

### 1.1 Loss distilasi

\[
\mathcal{L} = \alpha \cdot \mathcal{L}_{CE}(y, \hat{y}_{hard}) + (1-\alpha) \cdot \mathcal{L}_{MSE}(p_{teacher}, p_{student})
\]

- \(p_{teacher}\): probabilitas dari BERT (`Teacher.predict`)  
- \(\alpha\): hyperparameter `--alpha` (default 0.5)

### 1.2 Arsitektur

```
Teks  →  jieba + Tokenizer(Keras)  →  id sequence
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
    BERT (guru)            BiLSTM (murid)
    softmax logits         softmax + log_softmax
         │                     │
         └──── MSE soft ───────┘
               + CE hard
```

**Murid (`MiniModel`):** Embedding 256 → BiLSTM 256×2 → Linear → 2 kelas.

---

## 2. Data

Format `data/train.txt`, `dev.txt`, `test.txt`:

```text
0	Teks sampel kelas nol.
1	Teks sampel kelas satu.
```

Tab memisahkan label dan teks. Contoh disertakan di `data/`.

---

## 3. Langkah hands-on

### Langkah 1 — Latih guru BERT

```bash
cd Distillation/rnn_distill_bert
python train_bert.py
```

Simpan checkpoint ke `save_model/pytorch_model_epoch{N}.bin` (sesuaikan path di `Teacher`).

### Langkah 2 — Generate soft label (sekali)

```bash
python train_distill.py --is_need_knowledge True
```

Menyimpan:

- `knowledge/train` — softmax guru untuk train  
- `knowledge/dev` — untuk dev  

### Langkah 3 — Distilasi murid

```bash
python train_distill.py --is_need_knowledge False
```

Output: `save_model/distill_model_epoch{N}.bin`

### Parameter penting

| Argumen | Default | Arti |
|---------|---------|------|
| `--alpha` | 0.5 | Bobot CE vs MSE soft |
| `--teach_on_dev` | True | Update juga dari loss soft di dev |
| `--epochs` | 10 | Epoch distilasi |
| `--batch_size` | 2 | — |
| `--learning_rate` | 0.002 | LR murid (lebih besar dari BERT) |

---

## 4. Metrik evaluasi

| Metrik | Cara |
|--------|------|
| **Akurasi test** | Rata-rata batch `(pred == y)` — dicetak tiap epoch |
| **Loss distilasi** | CE + MSE — log per step |
| **F1** | Tambahkan `sklearn.f1_score` (opsional, guru pakai F1 di `train_bert.py`) |

**Perbandingan laporan:**

| Model | Akurasi | Parameter | Waktu inferensi |
|-------|---------|-----------|-----------------|
| BERT guru | — | ~110M | lambat |
| BiLSTM murid | — | ~jutaan | cepat |

---

## 5. Deploy

### 5.1 Inferensi murid

```python
import torch
from train_distill import MiniModel, load_data

# muat v_size dari tokenizer, muat state_dict distill_model_epoch9.bin
model = MiniModel(v_size)
model.load_state_dict(torch.load('save_model/distill_model_epoch9.bin', map_location='cpu'))
model.eval()

# x: LongTensor [batch, seq], lens: panjang sebenarnya
probs, log_probs = model(x, lens)
pred = probs.argmax(dim=1)
```

### 5.2 API ringkas

Murid cocok untuk **edge/CPU**: ukuran kecil, tanpa library `transformers` saat inferensi (cukup PyTorch + vocab Keras).

### 5.3 Catatan bahasa

Pipeline memakai **jieba** (Mandarin). Untuk Indonesia ganti tokenisasi (mis. spasi atau tokenizer IndoBERT) dan latih ulang.

---

## 6. Troubleshooting

| Masalah | Solusi |
|---------|--------|
| `knowledge/train` tidak ada | Jalankan dengan `--is_need_knowledge True` |
| Path epoch BERT salah | Edit `Teacher.__init__` baris `load_state_dict` |
| Akurasi murid rendah | Naikkan epoch; turunkan \(\alpha\) untuk lebih banyak soft label |

---

## Referensi

- Hinton et al., *Distilling the Knowledge in a Neural Network* (2015)  
- [shawroad/NLP_pytorch_project](https://github.com/shawroad/NLP_pytorch_project)
