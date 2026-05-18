# Hands-On: GlobalPointer (Span NER)

**Latih:** [`../GlobalPointer/run_train.py`](../GlobalPointer/run_train.py)  
**Inferensi:** [`../GlobalPointer/inference.py`](../GlobalPointer/inference.py)

---

## 1. Teori

**GlobalPointer** memprediksi matriks `(start, end)` per tipe entitas — setiap sel menandai apakah pasangan indeks token membentuk span entitas. Tidak perlu decoding BMES; cocok untuk **nested/overlap** dalam varian lanjutan.

Encoder: **RoBERTa** (`roberta_pretrain/`) + head pointer.

---

## 2. Data — CLUENER

Dataset [`data/`](../GlobalPointer/data/): `train.json`, `dev.json`, `test.json`, `ent2id.json`.

10 tipe entitas (Mandarin): address, book, company, game, government, movie, name, organization, position, scene.

Detail: [CLUENER](https://github.com/CLUEbenchmark/CLUENER) · lihat juga [`data/README.md`](../GlobalPointer/data/README.md).

---

## 3. Langkah hands-on

```bash
cd NER/GlobalPointer
# Unduh bobot RoBERTa ke roberta_pretrain/
pip install torch transformers tqdm
python run_train.py
```

Atau: `bash start.sh` (sesuaikan `CUDA_VISIBLE_DEVICES`).

---

## 4. Metrik

`utils.MetricsCalculator` — **F1, precision, recall** span-level pada validasi (rata-rata per batch di `run_train.py`).

Loss: `multilabel_categorical_crossentropy` untuk matriks pointer.

---

## 5. Deploy

```bash
python inference.py
```

- **API:** terima teks → kembalikan list span `(start, end, type)`  
- Bandingkan dengan modul BERT-CRF pada dataset yang sama (konversi format jika perlu)

---

## Referensi

- Su et al., GlobalPointer (2022)  
- [CLUEbenchmark/CLUENER](https://github.com/CLUEbenchmark/CLUENER)  
- [shawroad/NLP_pytorch_project](https://github.com/shawroad/NLP_pytorch_project)
