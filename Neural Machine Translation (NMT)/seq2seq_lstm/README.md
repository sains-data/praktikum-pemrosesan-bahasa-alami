# Hands-On: Seq2Seq LSTM + Attention

**Skrip:** [`../002-lstm_seq2seq_attention.py`](../002-lstm_seq2seq_attention.py)

---

## 1. Teori

Sama dengan varian GRU, tetapi **LSTM** memisahkan state sel (`c`) dan hidden (`h`). Attention menggabungkan embedding decoder dengan konteks encoder sebelum masuk ke LSTM.

Perbedaan utama vs GRU di kode ini: decoder memakai **cell state** encoder untuk inisialisasi attention.

---

## 2. Langkah hands-on

```bash
cd NMT
pip install torch keras numpy
python 002-lstm_seq2seq_attention.py
```

Data: [`../data/fin.txt`](../data/fin.txt).

---

## 3. Metrik & deploy

Sama seperti modul GRU: **loss**, **BLEU**, simpan checkpoint encoder/decoder untuk inferensi greedy.

Saat pelatihan selesai, skrip mencetak contoh **EN / terjemahan manusia / terjemahan mesin**.

---

## Referensi

- [shawroad/NLP_pytorch_project](https://github.com/shawroad/NLP_pytorch_project)
