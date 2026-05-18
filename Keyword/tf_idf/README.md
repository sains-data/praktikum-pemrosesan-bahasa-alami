# Hands-On: Ekstraksi Kata Kunci dengan TF-IDF

**Mata kuliah:** Pemrosesan Bahasa Alami Berbasis Transformer  
**Program Studi Sains Data · Fakultas Sains · Institut Teknologi Sumatera**

**Skrip:** [`../001-tf-idf提取关键词.py`](../001-tf-idf提取关键词.py)

---

## 1. Teori dan konsep

### 1.1 TF-IDF

- **TF (Term Frequency):** seberapa sering istilah \(t\) muncul dalam dokumen \(d\).  
- **IDF (Inverse Document Frequency):** \(\log \frac{N}{df(t)}\) — istilah langka di korpus mendapat bobot lebih tinggi.  
- **TF-IDF** = TF × IDF → kata yang **sering di dokumen ini** tetapi **jarang di dokumen lain** menjadi kandidat kuat kata kunci.

### 1.2 Implementasi jieba

`jieba.analyse.extract_tags` memperkirakan IDF dari korpus internal jieba dan menghitung skor per kata pada **satu dokumen** (mode default).

Filter POS `allowPOS` membatasi ke jenis kata bermakna (nama, verba, dll.).

---

## 2. Langkah hands-on

```bash
cd Keyword
pip install jieba
python 001-tf-idf提取关键词.py
```

Keluaran: daftar `(kata, skor)` — 20 teratas.

**Untuk teks Indonesia:** ganti isi `news.txt` atau ubah tokenisasi ke spasi + stopword list; jieba tetap bisa dipakai tetapi hasil optimal dengan model bahasa yang sesuai.

---

## 3. Metrik evaluasi

| Metrik | Kapan dipakai |
|--------|----------------|
| **Precision@K** | Dari K kata teratas, berapa yang benar menurut anotator |
| **Recall@K** | Proporsi kata kunci gold yang tertangkap |
| **F1@K** | Harmonic mean precision & recall |
| **Overlap antar metode** | Bandingkan dengan TextRank / KeyBERT pada dokumen sama |

Tanpa label gold, evaluasi **manual** (skala 1–5 relevansi) pada 10 dokumen sudah cukup untuk laporan praktikum.

---

## 4. Deploy

```python
import jieba.analyse

def kata_kunci_tfidf(teks, top_k=10):
    return jieba.analyse.extract_tags(
        teks, topK=top_k, withWeight=True,
        allowPOS=('n', 'nr', 'ns', 'nt', 'vn', 'v')
    )
```

**Batch:** loop file di folder → simpan CSV `(doc_id, keyword, score)`.

**API:** bungkus fungsi di FastAPI `POST /keywords` dengan body `{"text": "..."}`.

---

## Referensi

- Salton & McGill, *Introduction to Modern Information Retrieval*  
- [shawroad/NLP_pytorch_project](https://github.com/shawroad/NLP_pytorch_project)
