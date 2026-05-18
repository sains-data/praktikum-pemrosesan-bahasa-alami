# Hands-On: Seq2Seq GRU + Attention

**Skrip monolit:** [`../001-gru_seq2seq_attention.py`](../001-gru_seq2seq_attention.py)  
**Versi modular:** [`../GRU_attention/train.py`](../GRU_attention/train.py)

---

## 1. Teori

**Encoder–decoder** memetakan urutan sumber ke urutan target. **Attention** memungkinkan decoder “melihat” semua posisi encoder, bukan hanya vektor konteks terakhir.

```
EN tokens → BiGRU encoder → konteks per langkah
FR tokens ← Attention + BiGRU decoder ← teacher forcing (50%)
```

---

## 2. Data

- Monolit: [`../data/fin.txt`](../data/fin.txt) — format `kalimat_en\tkalimat_fr\t...`  
- Modular: [`../GRU_attention/data/fr2en.txt`](../GRU_attention/data/fr2en.txt) — `fr_tokens\ten_tokens`

---

## 3. Langkah hands-on

### Skrip tunggal (EN→FR)

```bash
cd NMT
pip install torch keras numpy
python 001-gru_seq2seq_attention.py
```

### Modul GRU_attention (FR→EN)

```bash
cd NMT/GRU_attention
pip install torch torchtext
python train.py
```

---

## 4. Metrik evaluasi

| Metrik | Cara |
|--------|------|
| **Loss NLL** | Dicetak per step saat latih |
| **BLEU** | `sacrebleu` atau `nltk.translate.bleu_score` pada set validasi |
| **chrF** | Alternatif untuk bahasa morfologis |

Contoh evaluasi BLEU (setelah decode greedy):

```bash
pip install sacrebleu
# bandingkan hipotesis vs referensi baris per baris
```

---

## 5. Deploy

- Simpan `encoder` + `decoder` state dict  
- Greedy / beam search decode untuk kalimat baru  
- **API:** `POST /translate` body `{"text": "...", "src": "en", "tgt": "fr"}`

---

## Referensi

- Bahdanau et al., Neural Machine Translation by Jointly Learning to Align and Translate (2015)  
- [shawroad/NLP_pytorch_project](https://github.com/shawroad/NLP_pytorch_project)
