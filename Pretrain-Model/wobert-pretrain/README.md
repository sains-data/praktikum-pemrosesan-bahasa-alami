# Hands-On: Pra-Pelatihan WoBERT

**Folder:** `wobert-pretrain/`

WoBERT memakai **whole-word masking** — seluruh kata (termasuk subword `##`) di-mask bersama, lebih natural untuk bahasa Mandarin.

---

## 1. Alur pipeline

```
Korpus: satu kalimat per baris, dokumen dipisah baris kosong
    → process_pretrain_data.py  → data/processed_data0.json
    → run_pretrain.py           → retrain_model/
```

Jalankan `process_pretrain_data.py` berkali-kali untuk **dynamic masking** (file JSON berbeda).

---

## 2. Instalasi

```bash
pip install torch transformers numpy
```

Unduh bobot WoBERT ke `wobert_pretrain/` (`pytorch_model.bin`, `bert_config.json`, `vocab.txt`).

---

## 3. Langkah hands-on

```bash
cd "Pretrain Model/wobert-pretrain"
python process_pretrain_data.py
python run_pretrain.py
```

Argumen CLI di `run_pretrain.py`: `--train_data_path`, `--batch_size`, `--epochs`, `--output_dir`.

---

## 4. Metrik evaluasi

Sama seperti BERT pretrain:

- **MLM accuracy** (`LMAccuracy`)  
- **NSP accuracy** (jika diaktifkan di loop)  
- **Loss** per epoch

---

## 5. Deploy

Output: `./retrain_model/` — gunakan untuk fine-tune tugas downstream Mandarin atau adaptasi ke domain serupa.

---

## Referensi

- Su et al., WoBERT (2020)  
- [shawroad/NLP_pytorch_project](https://github.com/shawroad/NLP_pytorch_project)
