# Hands-On: Ekstraksi Kata Kunci dengan TextRank

**Mata kuliah:** Pemrosesan Bahasa Alami Berbasis Transformer  
**Program Studi Sains Data · Fakultas Sains · Institut Teknologi Sumatera**

**Skrip:** [`../002-textrank提取关键词.py`](../002-textrank提取关键词.py)

---

## 1. Teori dan konsep

### 1.1 TextRank

Algoritma **berbasis graf** (mirip PageRank):

1. Tokenisasi + filter POS.  
2. Bangun graf: simpul = kata, tepi = co-occurrence dalam jendela.  
3. Iterasi skor PageRank pada graf → kata dengan skor tertinggi = kata kunci.

Berbeda dengan TF-IDF, TextRank mempertimbangkan **hubungan antar kata** dalam dokumen, bukan hanya frekuensi global.

### 1.2 Kapan cocok

- Satu dokumen panjang (berita, laporan).  
- Tidak butuh korpus besar (berbeda dengan IDF klasik multi-dokumen).  
- Kurang ideal jika dokumen sangat pendek (graf terlalu sparse).

---

## 2. Langkah hands-on

```bash
cd Keyword
pip install jieba
python 002-textrank提取关键词.py
```

Parameter sama dengan TF-IDF: `topK=20`, `withWeight=True`, `allowPOS=(...)`.

---

## 3. Metrik evaluasi

Gunakan metrik yang sama dengan modul TF-IDF (Precision@K, Recall@K, F1@K).

**Analisis tambahan untuk laporan:**

- Bandingkan urutan 10 kata teratas TF-IDF vs TextRank pada `news.txt`.  
- Jelaskan kata yang hanya muncul di salah satu metode (mis. kata umum vs entitas).

---

## 4. Deploy

```python
import jieba.analyse

def kata_kunci_textrank(teks, top_k=10):
    return jieba.analyse.textrank(
        teks, topK=top_k, withWeight=True,
        allowPOS=('n', 'nr', 'ns', 'nt', 'vn', 'v')
    )
```

Cocok untuk pipeline ringan tanpa GPU. Untuk bahasa Indonesia pertimbangkan `pytextrank` + model spaCy id.

---

## Referensi

- Mihalcea & Tarau, *TextRank: Bringing Order into Text* (2004)  
- [shawroad/NLP_pytorch_project](https://github.com/shawroad/NLP_pytorch_project)
