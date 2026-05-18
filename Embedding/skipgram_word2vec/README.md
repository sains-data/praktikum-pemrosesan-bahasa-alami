# Hands-On: Word2Vec Skip-gram

**Mata kuliah:** Pemrosesan Bahasa Alami Berbasis Transformer  
**Program Studi Sains Data · Fakultas Sains · Institut Teknologi Sumatera**

**Skrip:** [`../001-skipgram-word2vec.py`](../001-skipgram-word2vec.py)

---

## 1. Teori dan konsep

### 1.1 Distribusi semantik

Kata yang muncul dalam **konteks serupa** cenderung memiliki makna serupa (Harris, 1954). Word2Vec mempelajari vektor padat \( \mathbf{v}_w \in \mathbb{R}^d \) sehingga kata konteks prediktif mendekati kata pusat.

### 1.2 Skip-gram

Diberikan kata pusat \(w_c\), model memprediksi kata konteks \(w_o\) dalam jendela:

\[
P(w_o \mid w_c) = \text{softmax}(\mathbf{v}_{w_o}^\top \mathbf{v}_{w_c})
\]

Implementasi ini memakai **two-layer softmax** (matriks \(W\) dan \(W^T\)) dengan input one-hot.

### 1.3 Arsitektur kode

```
Kata pusat (one-hot) → W [V×d] → hidden [d] → W^T [d×V] → logits → CE loss
```

| Komponen | Ukuran demo |
|----------|-------------|
| `vocab_size` | Jumlah kata unik |
| `embedding_size` | 2 (untuk visualisasi 2D) |
| Jendela | 1 (kata kiri & kanan) |

---

## 2. Langkah hands-on

```bash
cd Embedding
pip install torch numpy matplotlib
python 001-skipgram-word2vec.py
```

1. Korpus contoh Inggris kecil dibangun otomatis.  
2. Pasangan skip-gram `(pusat, konteks)` dibuat.  
3. Latih 5000 epoch; loss dicetak tiap 1000 epoch.  
4. Plot 2D: setiap kata sebagai titik dengan label.

**Eksperimen:** ubah `embedding_size` ke 50–100 pada korpus lebih besar; hilangkan `plt.show()` jika headless.

---

## 3. Metrik evaluasi

| Metrik | Cara |
|--------|------|
| **Training loss (CE)** | Harus turun; stabil di akhir |
| **Visualisasi 2D** | Kata serupa (dog/cat/animal) berdekatan |
| **Analogi (opsional)** | `king - man + woman ≈ queen` dengan `gensim` pada skala besar |
| **Similarity kosinus** | Antara pasangan kata yang diharapkan mirip |

Untuk laporan: screenshot plot + 3 pasangan kata dengan jarak kosinus.

---

## 4. Deploy / penggunaan embedding

### 4.1 Ekstrak vektor

```python
W, WT = model.parameters()
vec_dog = W[vocab2id['dog']].detach().numpy()
```

### 4.2 Produksi

- Latih Word2Vec/FastText offline (`gensim`, `word2vec`).  
- Simpan `KeyedVectors` → lookup O(1) saat inferensi.  
- Untuk bahasa Indonesia: korpus besar + tokenisasi yang konsisten.

### 4.3 Keterbatasan

- **Tidak kontekstual** — satu vektor per kata (beda dengan BERT).  
- Kata jarang (OOV) perlu `<UNK>` atau subword.

---

## Referensi

- Mikolov et al., *Efficient Estimation of Word Representations in Vector Space* (2013)  
- [shawroad/NLP_pytorch_project](https://github.com/shawroad/NLP_pytorch_project)
