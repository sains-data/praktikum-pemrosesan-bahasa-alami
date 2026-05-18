# NMT — Neural Machine Translation

**Mata kuliah:** Pemrosesan Bahasa Alami Berbasis Transformer  
**Program Studi Sains Data · Fakultas Sains · Institut Teknologi Sumatera**

Modul ini membandingkan **Neural Machine Translation** dari arsitektur klasik seq2seq hingga Transformer.

| Pendekatan | Lokasi | Bahasa contoh |
|------------|--------|----------------|
| GRU + Attention (monolit) | [`001-gru_seq2seq_attention.py`](001-gru_seq2seq_attention.py) | Inggris → Prancis |
| LSTM + Attention (monolit) | [`002-lstm_seq2seq_attention.py`](002-lstm_seq2seq_attention.py) | Inggris → Prancis |
| GRU + Attention (modular) | [`GRU_attention/`](GRU_attention/) | Prancis → Inggris |
| **Transformer** | [`Transformers_NMT/`](Transformers_NMT/) | Mandarin → Inggris |

| Panduan | Tautan |
|---------|--------|
| Seq2seq GRU | [seq2seq_gru/README.md](seq2seq_gru/README.md) |
| Seq2seq LSTM | [seq2seq_lstm/README.md](seq2seq_lstm/README.md) |
| Transformer NMT | [transformer_nmt/README.md](transformer_nmt/README.md) |

**Sumber asli:** [shawroad/NLP_pytorch_project](https://github.com/shawroad/NLP_pytorch_project) — folder `NMT`.

---

## Alur belajar

1. **GRU/LSTM + attention** — pahami encoder–decoder, teacher forcing, loss NLL.  
2. **GRU_attention/train.py** — versi modular dengan `torchtext`.  
3. **Transformers_NMT** — self-attention, masked decoder, implementasi “Attention Is All You Need”.

---

## Data

| Berkas | Pasangan bahasa |
|--------|-----------------|
| `data/fin.txt` | EN–FR (tab-separated) |
| `GRU_attention/data/fr2en.txt` | FR → EN |
| `Transformers_NMT/data/train.zh` + `train.en` | ZH → EN |
