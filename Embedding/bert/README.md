# Hands-On: BERT — Implementasi Pra-pelatihan dari Nol

**Mata kuliah:** Pemrosesan Bahasa Alami Berbasis Transformer  
**Program Studi Sains Data · Fakultas Sains · Institut Teknologi Sumatera**

**Skrip:** [`../002-bert.py`](../002-bert.py)

---

## 1. Teori dan konsep

### 1.1 BERT (Bidirectional Encoder Representations from Transformers)

BERT adalah **encoder Transformer** yang dilatih dengan dua tugas pra-pelatihan:

1. **Masked Language Model (MLM)** — prediksi token yang di-mask (~15%).  
2. **Next Sentence Prediction (NSP)** — apakah kalimat B mengikuti kalimat A.

### 1.2 Arsitektur dalam skrip

```
Input IDs + Segment IDs
        ↓
Embedding (token + posisi + segmen) + LayerNorm
        ↓
× n_layers EncoderLayer (Multi-Head Attention + FFN)
        ↓
    ┌───┴───┐
    ↓       ↓
  MLM     NSP (dari token [CLS])
```

| Hyperparameter demo | Nilai |
|---------------------|-------|
| `n_layers` | 6 |
| `n_heads` | 12 |
| `d_model` | 768 |
| `d_ff` | 3072 |
| `maxlen` | 30 |

### 1.3 Masking (strategi BERT)

- 80% → `[MASK]`  
- 10% → token acak  
- 10% → tetap asli  

---

## 2. Langkah hands-on

```bash
cd Embedding
pip install torch numpy
python 002-bert.py
```

Alur internal:

1. Korpus dialog Romeo–Juliet (teks contoh).  
2. `make_batch()` — pasangan kalimat + MLM + label NSP.  
3. Satu batch dilatih 100 epoch (demo edukatif; produksi butuh jutaan langkah).  
4. Inferensi pada satu sampel batch: token ter-mask dan prediksi `isNext`.

---

## 3. Metrik evaluasi

| Metrik | Arti |
|--------|------|
| `loss_lm` | Cross-entropy prediksi token ter-mask |
| `loss_clsf` | CE tugas NSP (2 kelas) |
| `loss` total | `loss_lm + loss_clsf` |
| **Akurasi MLM** (tambahan) | % token mask benar |
| **Akurasi NSP** | % pasangan kalimat benar |

**Checklist laporan:**

- [ ] Kurva loss total vs epoch  
- [ ] Contoh prediksi MLM (sebelum/sesudah latih)  
- [ ] Prediksi NSP vs label `isNext`  

Pada skala penuh: evaluasi **GLUE**, **perplexity** pada held-out corpus.

---

## 4. Deploy

### 4.1 Pra-pelatihan skala penuh

Skrip ini untuk **pemahaman**; produksi memakai:

```bash
# Hugging Face (disarankan untuk deploy)
from transformers import BertModel, BertTokenizer
model = BertModel.from_pretrained('bert-base-multilingual-cased')
```

### 4.2 Fine-tuning downstream

Setelah pra-latih (atau muat checkpoint HF):

1. Ganti head NSP/MLM dengan head klasifikasi/NER.  
2. Latih pada data berlabel Itera.  
3. Simpan `state_dict` + tokenizer.

### 4.3 Serving embedding

```python
outputs = model(**inputs)
cls_embedding = outputs.last_hidden_state[:, 0, :]  # vektor [CLS]
```

---

## 5. Troubleshooting

| Masalah | Solusi |
|---------|--------|
| Loss tidak turun | Normal pada 1 batch kecil; tambah data & epoch |
| OOM | Kurangi `d_model`, `n_layers`, `batch_size` |
| `[PAD]` di vocab | Token padding khusus demo asli |

---

## Referensi

- Devlin et al., *BERT* (2018)  
- Vaswani et al., *Attention Is All You Need* (2017)  
- [shawroad/NLP_pytorch_project](https://github.com/shawroad/NLP_pytorch_project)
