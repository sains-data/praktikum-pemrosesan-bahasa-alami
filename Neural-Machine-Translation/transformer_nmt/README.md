# Hands-On: Transformer untuk Terjemahan Mesin

**Folder kode:** [`../Transformers_NMT/`](../Transformers_NMT/)  
**Latih:** [`train.py`](../Transformers_NMT/train.py)  
**Pra-pemrosesan:** [`process_data.py`](../Transformers_NMT/process_data.py)

---

## 1. Teori

Arsitektur **Transformer** (Vaswani et al., 2017):

- **Encoder:** self-attention + FFN, positional encoding  
- **Decoder:** masked self-attention + cross-attention ke encoder  
- **Loss:** cross-entropy dengan label smoothing (`transformer/loss.py`)

Pasangan bahasa contoh: **Mandarin → Inggris** (file paralel `train.zh` / `train.en`).

---

## 2. Instalasi

```bash
pip install torch jieba nltk tqdm tensorboard
python -c "import nltk; nltk.download('punkt')"
```

---

## 3. Langkah hands-on

### Langkah 1 — Bangun kosakata & pickle

```bash
cd NMT/Transformers_NMT
python process_data.py
```

Menghasilkan `data/vocab.pkl` dan `data/data.pkl`.

### Langkah 2 — Latih

```bash
python train.py
# atau: bash start.sh
```

Log TensorBoard: folder `runs/`.

### Langkah 3 — Uji sampel

```bash
python data_gen.py
```

---

## 4. Konfigurasi

[`config.py`](../Transformers_NMT/config.py):

| Parameter | Default | Fungsi |
|-----------|---------|--------|
| `n_src_vocab` / `n_tgt_vocab` | 15000 | Ukuran kosakata |
| `maxlen_in` / `maxlen_out` | 50 / 100 | Filter panjang |
| `d_model` | 512 | Dimensi model |
| `pad_id`, `sos_id`, `eos_id` | 0, 1, 2 | Token khusus |

Hyperparameter CLI: `utils.parse_args()` di `train.py` (layer, head, dropout, epoch).

---

## 5. Metrik evaluasi

| Metrik | Implementasi |
|--------|----------------|
| **Loss / perplexity** | `cal_performance` pada validasi |
| **BLEU** | Decode valid set → `sacrebleu` vs `valid.en` |
| **TensorBoard** | Kurva loss per epoch |

**Checklist laporan:**

- [ ] Kurva train vs valid loss  
- [ ] 5 contoh terjemahan ZH→EN  
- [ ] BLEU pada `valid` (bandingkan dengan seq2seq GRU jika dilatih)

---

## 6. Deploy

1. Simpan checkpoint terbaik (`utils.save_checkpoint`).  
2. Muat encoder–decoder, jalankan decode autoregressive dengan `sos_id` / `eos_id`.  
3. Produksi: gunakan **MarianMT**, **NLLB**, atau **mBART** dari Hugging Face; modul ini untuk pembelajaran implementasi dari nol.

---

## 7. Data Bahasa Indonesia

Ganti pasangan file paralel, mis. `train.id` + `train.en`, sesuaikan tokenisasi di `process_data.py` (jieba → tokenizer ID).

---

## Referensi

- Vaswani et al., *Attention Is All You Need* (2017)  
- [shawroad/NLP_pytorch_project](https://github.com/shawroad/NLP_pytorch_project)
