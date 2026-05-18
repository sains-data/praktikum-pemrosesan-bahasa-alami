# Hands-On: Pra-Pelatihan BERT (MLM + NSP)

**Folder:** `bert-pretrain/`

---

## 1. Teori

**BERT pre-training** memiliki dua tugas:

1. **Masked Language Model (MLM)** — prediksi token yang di-mask (15% token).  
2. **Next Sentence Prediction (NSP)** — apakah kalimat B mengikuti kalimat A.

Continued pre-training melanjutkan bobot `bert-base` pada korpus Anda agar distribusi bahasa domain lebih cocok.

---

## 2. Alur pipeline

```
corpus/corpus.txt     → pro_data.py → pro_data.txt (per kalimat, dokumen dipisah baris kosong)
                      → get_train_data.py → process_data0.json (token + mask + NSP)
                      → run_pretrain.py → checkpoint
```

---

## 3. Instalasi

```bash
pip install torch transformers
```

Siapkan `bert_pretrain/bert_config.json`, `vocab.txt`, dan bobot awal (unduh BERT-base Chinese atau multilingual).

---

## 4. Langkah hands-on

```bash
cd "Pretrain Model/bert-pretrain"

# 1. Satu artikel per baris di corpus/corpus.txt
python corpus/pro_data.py

# 2. Bangun sampel MLM+NSP
python get_train_data.py

# 3. Latih
python run_pretrain.py
```

Fine-tune downstream (contoh): `finetuning_task_demo.py`.

---

## 5. Metrik evaluasi

| Metrik | Di kode | Arti |
|--------|---------|------|
| **MLM accuracy** | `LMAccuracy` / `tr_mask_acc` | Token ter-mask diprediksi benar |
| **NSP accuracy** | `tr_sop_acc` | Klasifikasi urutan kalimat benar |
| **Train loss** | `tr_loss` | Cross-entropy gabungan |
| **Perplexity** | `exp(loss)` | Turun saat pelatihan stabil |

**Checklist laporan:**

- [ ] Kurva MLM acc vs step  
- [ ] Bandingkan fine-tune task sebelum/sesudah continued pretrain  
- [ ] Ukuran korpus vs gain validasi

---

## 6. Deploy

Checkpoint `pytorch_model.bin` + `config.json` + `vocab.txt` → muat dengan:

```python
from transformers import BertForPreTraining, BertTokenizer
model = BertForPreTraining.from_pretrained("./output_checkpoint")
```

Untuk produksi task spesifik, lanjutkan ke `BertForSequenceClassification.from_pretrained(..., state_dict=...)`.

---

## Referensi

- Devlin et al., BERT (2018)  
- [shawroad/NLP_pytorch_project](https://github.com/shawroad/NLP_pytorch_project)
