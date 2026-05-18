# Hands-On: ALBERT — Embedding Faktorisasi & Berbagi Bobot

**Mata kuliah:** Pemrosesan Bahasa Alami Berbasis Transformer  
**Program Studi Sains Data · Fakultas Sains · Institut Teknologi Sumatera**

**Skrip:** [`../003-albert.py`](../003-albert.py)

---

## 1. Teori dan konsep

### 1.1 Motivasi ALBERT

BERT besar memakan memori dan waktu latih. **ALBERT** (Lan et al., 2019) mengurangi parameter dengan:

1. **Factorized embedding** — embedding vocabulary kecil \(E \ll H\), lalu proyeksi ke hidden \(H\).  
2. **Parameter sharing** — satu blok attention + FFN dipakai ulang untuk semua lapis.  
3. **SOP** (Sentence Order Prediction) — di paper asli menggantikan NSP; demo ini tetap memakai NSP seperti BERT untuk kesederhanaan.

### 1.2 Perbedaan dengan `002-bert.py`

| Aspek | BERT (`002`) | ALBERT (`003`) |
|-------|--------------|----------------|
| Embedding | `Embedding(vocab, d_model)` | `Embedding(vocab, E)` + `Linear(E, H)` |
| Lapis encoder | `ModuleList` terpisah | Satu `attn` + `pwff` diulang `n_layers` |
| Ukuran demo | `d_model=768` | `embedding=24`, `hidden=64` |

### 1.3 Arsitektur

```
token id → tok_embed1 [E] → tok_embed2 [H] + pos + seg
                ↓
    for _ in range(n_layers):  # bobot sama tiap iterasi
        MultiHeadAttention → residual + LayerNorm
        FFN → residual + LayerNorm
                ↓
         MLM decoder (berbagi bobot embedding) + NSP head
```

---

## 2. Langkah hands-on

```bash
cd Embedding
python 003-albert.py
```

Sama seperti modul BERT: korpus Romeo–Juliet, `make_batch()`, 100 epoch, cetak prediksi MLM & NSP.

**Perhatikan di kode:** kelas `Embeddings` — `tok_embed1` dan `tok_embed2` (faktorisasi).

---

## 3. Metrik evaluasi

| Metrik | Keterangan |
|--------|------------|
| `loss_lm` + `loss_clsf` | Sama seperti BERT demo |
| **Jumlah parameter** | Bandingkan `sum(p.numel())` ALBERT vs BERT pada dimensi sama |
| **Kecepatan langkah** | ALBERT lebih sedikit bobot unik |

```python
total = sum(p.numel() for p in model.parameters())
print('Parameter:', total)
```

---

## 4. Deploy

- Produksi: `transformers.AlbertModel.from_pretrained(...)`.  
- Pilih varian `albert-base-v2` / multilingual sesuai bahasa.  
- Fine-tuning sama seperti BERT; head downstream identik secara konsep.

**Kapan memilih ALBERT:** resource terbatas, masih butuh kualitas mendekati BERT.

---

## Referensi

- Lan et al., *ALBERT: A Lite BERT* (2019)  
- [shawroad/NLP_pytorch_project](https://github.com/shawroad/NLP_pytorch_project)
