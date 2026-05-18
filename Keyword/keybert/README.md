# Hands-On: Ekstraksi Kata Kunci dengan KeyBERT

**Mata kuliah:** Pemrosesan Bahasa Alami Berbasis Transformer  
**Program Studi Sains Data · Fakultas Sains · Institut Teknologi Sumatera**

**Skrip:** [`../003-keybert提取关键词.py`](../003-keybert提取关键词.py)

---

## 1. Teori dan konsep

### 1.1 KeyBERT

KeyBERT memakai **embedding kontekstual** (BERT dan sejenisnya):

1. Embedding dokumen penuh → vektor dokumen.  
2. Embedding setiap kandidat kata/frasa → vektor kandidat.  
3. **Cosine similarity** antara dokumen dan kandidat → skor relevansi.

Kata kunci adalah istilah yang **paling mirip maknanya** dengan keseluruhan teks dalam ruang semantik BERT.

### 1.2 Perbedaan dengan TF-IDF / TextRank

| Aspek | TF-IDF / TextRank | KeyBERT |
|-------|-------------------|---------|
| Representasi | Statistik / graf | Transformer |
| Sinonim | Terpisah | Lebih dekat dalam embedding |
| GPU | Tidak wajib | Disarankan |
| Bahasa | jieba (ZH) | Pilih model (`bert-base-chinese`, multilingual) |

---

## 2. Persiapan

```bash
pip install keybert jieba sentence-transformers torch
```

Unduh model pertama kali (butuh internet): `bert-base-chinese` atau ganti ke:

```python
model = KeyBERT('paraphrase-multilingual-MiniLM-L12-v2')  # lebih cocok multibahasa
```

---

## 3. Langkah hands-on

```bash
cd Keyword
python 003-keybert提取关键词.py
```

- Teks dibaca dari `news.txt`.  
- Tokenisasi jieba → string dipisah spasi (untuk kandidat n-gram).  
- `keyphrase_ngram_range=(1, 2)` → unigram dan bigram.  
- `top_n=20` kata kunci.

---

## 4. Metrik evaluasi

| Metrik | Cara |
|--------|------|
| **Precision@K / Recall@K** | Bandingkan dengan anotasi manual |
| **Cosine score** | KeyBERT mengembalikan skor similarity — gunakan sebagai confidence |
| **Waktu inferensi** | Catat detik/dokumen vs TF-IDF |
| **Studi perbandingan** | Tabel 3 metode × 5 dokumen |

---

## 5. Deploy

```python
from keybert import KeyBERT
import jieba

model = KeyBERT('paraphrase-multilingual-MiniLM-L12-v2')

def kata_kunci_keybert(teks, top_n=10):
    doc = ' '.join(jieba.cut(teks))  # atau tokenizer Indonesia
    return model.extract_keywords(
        doc, keyphrase_ngram_range=(1, 2), top_n=top_n
    )
```

**Produksi:**

- Cache model di memori worker (jangan load ulang per request).  
- Batasi panjang teks (truncate 512 token).  
- GPU serving dengan batch kecil untuk throughput.

**Docker:** image dengan `sentence-transformers` + model pre-downloaded.

---

## 6. Troubleshooting

| Masalah | Solusi |
|---------|--------|
| Download model gagal | Set proxy/HF mirror; unduh manual |
| OOM | Model MiniLM lebih kecil |
| Kata kunci tidak relevan (ID) | Ganti model multilingual; kurangi n-gram |

---

## Referensi

- Grootendorst, *KeyBERT: Minimal keyword extraction with BERT* (2020)  
- [shawroad/NLP_pytorch_project](https://github.com/shawroad/NLP_pytorch_project)
