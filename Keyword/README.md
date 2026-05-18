# Ekstraksi Kata Kunci (Keyword Extraction)

**Mata kuliah:** Pemrosesan Bahasa Alami Berbasis Transformer  
**Program Studi Sains Data · Fakultas Sains · Institut Teknologi Sumatera**

Modul ini membandingkan tiga pendekatan mengekstrak **kata kunci** dari dokumen teks: statistik klasik (TF-IDF), graf (TextRank), dan embedding Transformer (KeyBERT).

| Metode | Skrip | Panduan |
|--------|--------|---------|
| TF-IDF | `001-tf-idf提取关键词.py` | [tf_idf/README.md](tf_idf/README.md) |
| TextRank | `002-textrank提取关键词.py` | [textrank/README.md](textrank/README.md) |
| KeyBERT | `003-keybert提取关键词.py` | [keybert/README.md](keybert/README.md) |

**Data contoh:** [`news.txt`](news.txt) (berita Mandarin; ganti dengan artikel Bahasa Indonesia untuk praktikum).

**Sumber asli:** [shawroad/NLP_pytorch_project](https://github.com/shawroad/NLP_pytorch_project) — folder `Keyword`.
