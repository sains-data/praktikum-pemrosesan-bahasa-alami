# Hands-On: NPLM — Neural Probabilistic Language Model

**Mata kuliah:** Pemrosesan Bahasa Alami Berbasis Transformer  
**Program Studi Sains Data · Fakultas Sains · Institut Teknologi Sumatera**

**Skrip:** [`../004-NPLM.py`](../004-NPLM.py)

---

## 1. Teori dan konsep

### 1.1 Sejarah

**NPLM** (Bengio et al., 2003) adalah salah satu model neural pertama yang mempelajari **embedding kata** dalam layer tersembunyi untuk memprediksi kata berikutnya dari konteks tetap.

### 1.2 Model dalam skrip

Diberikan **2 kata konteks** \((w_{t-2}, w_{t-1})\), prediksi kata \(w_t\):

```
w_{t-2}, w_{t-1}  →  Embedding  →  concat  →  Linear(128)  →  ReLU  →  Linear(V)  →  log_softmax
```

| Komponen | Nilai demo |
|----------|------------|
| `context_size` | 2 |
| `embedding_dim` | 128 |
| Loss | NLLLoss (setara CE pada log-prob) |

### 1.3 Data

- File: `data/corpus.txt` (UTF-8, novel atau teks panjang).  
- Tokenisasi: **jieba** (cocok Mandarin; untuk Indonesia bisa diganti spasi/`nltk`).

---

## 2. Persiapan

```bash
pip install torch jieba numpy
mkdir -p Embedding/data
# Salin file teks UTF-8 ke Embedding/data/corpus.txt
```

---

## 3. Langkah hands-on

```bash
cd Embedding
python 004-NPLM.py
```

1. Baca korpus → token (jieba) → filter tanda baca.  
2. Bangun **trigram**: `([w_i, w_{i+1}], w_{i+2})`.  
3. `DataLoader` batch 64, Adam lr=0.001, 20 epoch.  
4. Simpan kurva loss ke `loss.json`.  
5. Ekstrak semua vektor embedding via `model.extract()`.

---

## 4. Metrik evaluasi

| Metrik | Cara |
|--------|------|
| **Loss NLL per epoch** | Rata-rata loss batch (dicetak) |
| **Perplexity** | `exp(loss_rata_rata)` — semakin rendah semakin baik |
| **Similarity** | Kosinus antar vektor kata yang relevan |

**Contoh perplexity:**

```python
import math
ppl = math.exp(rata_rata_loss)
print('Perplexity:', ppl)
```

---

## 5. Deploy

### 5.1 Lookup embedding

```python
ids = torch.LongTensor([[vocab2id['kata1'], vocab2id['kata2']]])
vec = model.extract(ids)  # [1, 2, embedding_dim]
```

### 5.2 Keterbatasan produksi

NPLM konteks pendek dan lambat pada vocab besar — digantikan Word2Vec dan Transformer. Tetap penting secara **pedagogis** sebagai jembatan ke embedding neural.

### 5.3 Bahasa Indonesia

Ganti `jieba.lcut` dengan:

```python
words = text.lower().split()  # baseline
# atau tokenizer IndoNLU / Hugging Face
```

---

## 6. Troubleshooting

| Masalah | Solusi |
|---------|--------|
| `FileNotFoundError` corpus | Buat `data/corpus.txt` |
| Loss NaN | Kurangi lr; periksa kosakata kosong |
| Pelatihan lambat | Kurangi ukuran korpus untuk uji |

---

## Referensi

- Bengio et al., *A Neural Probabilistic Language Model* (2003)  
- [shawroad/NLP_pytorch_project](https://github.com/shawroad/NLP_pytorch_project)
