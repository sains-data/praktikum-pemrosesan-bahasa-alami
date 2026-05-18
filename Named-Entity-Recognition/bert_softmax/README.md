# Hands-On: BERT + Softmax untuk NER

**Latih:** [`../Bert_Softmax_Ner/train.py`](../Bert_Softmax_Ner/train.py)  
**Inferensi:** [`../Bert_Softmax_Ner/inference.py`](../Bert_Softmax_Ner/inference.py)  
**Pra-pemrosesan data:** [`../Bert_Softmax_Ner/data_helper.py`](../Bert_Softmax_Ner/data_helper.py)

---

## 1. Teori

Setiap token diklasifikasi ke salah satu label BIO (7 kelas pada MSRA). **CrossEntropyLoss** dengan `ignore_index=0` untuk padding. Lebih sederhana dari CRF; tidak memaksa transisi label valid.

---

## 2. Data

- Folder `data/msra数据集` — unduh dataset MSRA, jalankan `data_helper.py` untuk membuat `train/val/test` + `tags.txt`  
- Pra-latih BERT: `bert_pretrain/` (`config.json`, `pytorch_model.bin`)

---

## 3. Langkah hands-on

```bash
cd NER/Bert_Softmax_Ner
python data_helper.py    # sekali: siapkan split
python train.py
python inference.py
```

Konfigurasi: [`config.py`](../Bert_Softmax_Ner/config.py) — `batch_size`, `max_len`, `epoch_num`.

---

## 4. Metrik

- **Loss validasi** — early stopping di `train.py` (`best_val_f1`)  
- Tambahkan **seqeval** untuk entity-level F1 pada laporan:

```bash
pip install seqeval
```

---

## 5. Deploy

Checkpoint: `save_model/best_model.bin`. Mapping tag: `data/msra/tags.txt`.

Untuk Bahasa Indonesia: ganti tokenizer/checkpoint (mis. IndoBERT) dan data berformat `kata\tlabel` per baris.

---

## Referensi

- [shawroad/NLP_pytorch_project](https://github.com/shawroad/NLP_pytorch_project)
