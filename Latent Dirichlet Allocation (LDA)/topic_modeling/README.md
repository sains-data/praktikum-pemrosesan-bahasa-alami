# Hands-On: Pemodelan Topik dengan LDA

**Mata kuliah:** Pemrosesan Bahasa Alami Berbasis Transformer  
**Program Studi Sains Data · Fakultas Sains · Institut Teknologi Sumatera**

**Skrip:** [`../run.py`](../run.py)

---

## 1. Teori dan konsep

### 1.1 Apa itu LDA?

**Latent Dirichlet Allocation (LDA)** adalah model probabilistik yang mengasumsikan:

- Setiap **dokumen** adalah campuran beberapa **topik**.  
- Setiap **topik** adalah distribusi probabilitas atas **kata**.  

Secara intuitif: jika banyak dokumen membahas “mahasiswa + belajar + perpustakaan”, LDA dapat menemukan topik yang diwakili kata-kata seperti `belajar`, `perpustakaan`, `ujian`, dll.

### 1.2 Alur pipeline dalam `run.py`

```
answers.csv (teks mentah)
        ↓
Pembersihan + tokenisasi (jieba)
        ↓
TF-IDF (sklearn TfidfVectorizer)
        ↓
LDA (LatentDirichletAllocation)
        ↓
top_vocab.csv  +  result.csv
```

| Tahap | Library | Output |
|-------|---------|--------|
| Pra-pemrosesan | `jieba`, `re` | Kolom `cut` |
| Vektorisasi | `TfidfVectorizer` | Matriks dokumen–kata |
| Topik | `LatentDirichletAllocation` | Kata per topik + P(topik\|dokumen) |

### 1.3 Perbedaan dengan clustering teks

| LDA | K-means pada embedding |
|-----|------------------------|
| Topik = distribusi kata | Klaster = grup dokumen |
| Interpretable via kata kunci | Perlu embedding dulu |
| Butuh pilih \(K\) topik | Butuh pilih \(K\) klaster |

---

## 2. Data contoh

File [`answers.csv`](../answers.csv): jawaban Zhihu (Mandarin) dengan kolom antara lain `回答内容` (isi jawaban).

Untuk korpus Indonesia, siapkan CSV dengan kolom teks, misalnya:

```csv
id,isi_teks
1,"Artikel atau jawaban panjang dalam Bahasa Indonesia ..."
```

Lalu ubah di `run.py`:

```python
document_column_name = 'isi_teks'
```

---

## 3. Langkah hands-on

### Langkah 1 — Dependensi

```bash
pip install pandas numpy scikit-learn jieba
```

### Langkah 2 — Jalankan pipeline

```bash
cd LDA
python run.py
```

### Langkah 3 — Interpretasi hasil

**`top_vocab.csv`** — setiap baris = satu topik; setiap kolom = kata dengan bobot tinggi pada topik tersebut (20 kata teratas).

**`result.csv`** — setiap baris = satu dokumen; kolom `P(topic k)` = probabilitas dokumen termasuk topik \(k\).

Contoh: jika `P(topic 5) ≈ 0.96`, dokumen tersebut sangat dominan pada topik 5.

### Langkah 4 — Tuning hyperparameter

Edit di `run.py`:

| Parameter | Default | Saran |
|-----------|---------|--------|
| `n_topics` | 5 | Coba 3–15; terlalu banyak → topik tumpang tindih |
| `n_top_words` | 20 | Kata representatif per topik |
| `max_iter` | 50 | Naikkan jika belum konvergen |
| `learning_method` | `'online'` | Cocok data besar |

**Memilih jumlah topik \(K\):**

- Coherence score (gensim `CoherenceModel`)  
- Perplexity (semakin rendah semakin baik, dengan hati-hati)  
- Uji coba visual + label manual topik

---

## 4. Metrik evaluasi

| Metrik | Arti | Cara |
|--------|------|------|
| **Topic coherence (c_v, u_mass)** | Kata dalam topik sering co-occur | `gensim.models.CoherenceModel` |
| **Perplexity** | Seberapa baik model prediksi data hold-out | `lda.perplexity(tf_idf)` |
| **Diversity topik** | Overlap kata antar topik | Hitung intersection top-10 words |
| **Evaluasi manual** | Labeli 5 topik dengan nama (mis. “pendidikan”, “relasi”) | Wajib untuk laporan |

**Checklist laporan:**

- [ ] Tabel 5 topik × 10 kata teratas  
- [ ] 3 contoh dokumen + distribusi P(topik)  
- [ ] Justifikasi pemilihan `n_topics=5`  
- [ ] Bandingkan dengan TextRank/KeyBERT dari modul Keyword (satu dokumen vs korpus)

---

## 5. Deploy

### 5.1 Fungsi inferensi topik dokumen baru

```python
import joblib
# Setelah fit sekali, simpan:
# joblib.dump({'lda': lda, 'vectorizer': tf_idf_vectorizer}, 'lda_model.pkl')

bundle = joblib.load('lda_model.pkl')
X_new = bundle['vectorizer'].transform([teks_tokenized])
proba = bundle['lda'].transform(X_new)[0]
topik_dominan = proba.argmax()
```

### 5.2 Batch & API

- **Batch:** cron harian pada arsip artikel → update dashboard topik trending.  
- **FastAPI:** `POST /topics` dengan body teks → kembalikan distribusi topik + kata kunci.

### 5.3 Produksi

- Latih offline; serving hanya `transform` (cepat).  
- Untuk bahasa Indonesia: ganti jieba dengan tokenizer ID + stopword.  
- Pertimbangkan **BERTopic** (embedding + clustering) jika butuh topik lebih koheren secara semantik.

---

## 6. Troubleshooting

| Masalah | Solusi |
|---------|--------|
| `KeyError: 回答内容` | Sesuaikan `document_column_name` dengan header CSV |
| Topik tidak terbaca | Ubah `random_state`, tambah `max_iter`, kurangi `n_topics` |
| Semua dokumen satu topik dominan | Normal jika korpus homogen; tambah variasi data atau preprocessing |
| Memory error | `TfidfVectorizer(max_features=5000)` |

---

## Referensi

- Blei et al., *Latent Dirichlet Allocation* (2003)  
- [sklearn LDA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html)  
- [shawroad/NLP_pytorch_project](https://github.com/shawroad/NLP_pytorch_project)
