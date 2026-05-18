# Hands-On: Pra-Pelatihan Mask-Only (MLM saja)

**Skrip utama:** [`run_pretrain.py`](run_pretrain.py)  
**Data:** [`data/text.txt`](data/text.txt) — **satu kalimat per baris**

---

## 1. Teori

Beberapa praktisi menganggap **NSP kurang penting** untuk continued pretraining; cukup melatih **Masked LM** pada kalimat domain. Modul ini memakai `BertForPreTraining` tetapi hanya memakai loss pada `prediction_logits` (mask), bukan `seq_relationship_logits`.

Cocok untuk korpus Mandarin pendek per baris atau kalimat hasil segmentasi.

---

## 2. Instalasi

```bash
pip install torch transformers tensorboardX
```

Bobot awal di `roberta_pretrain/` (`config.json`, `pytorch_model.bin`, `vocab.txt`).

---

## 3. Langkah hands-on

```bash
cd "Pretrain Model/mask_pretrain"
# Edit data/text.txt — satu kalimat per baris

python run_pretrain.py \
  --train_data_path ./data/text.txt \
  --pretrain_weight ./roberta_pretrain \
  --output_dir ./outputs
```

Log TensorBoard: folder `log/log`.

---

## 4. Konfigurasi

[`config.py`](config.py) — epoch, batch size, learning rate, `save_checkponint_steps`, akumulasi gradien.

---

## 5. Metrik evaluasi

| Metrik | Implementasi |
|--------|----------------|
| **MLM accuracy** | `get_metric_acc()` — proporsi token mask benar (abaikan padding/id 0) |
| **Loss** | CrossEntropy pada posisi ter-mask |
| **TensorBoard** | `train loss`, `train accuracy` tiap 100 step |

---

## 6. Deploy

Checkpoint disimpan di `outputs/pytorch_model_epoch{step}.bin` beserta `config.json` dan vocab.

Gunakan bobot untuk fine-tune:

```python
from transformers import BertForSequenceClassification
# Muat encoder dari checkpoint pra-latih, lalu ganti head klasifikasi
```

---

## 7. Data Bahasa Indonesia

Ganti `text.txt` dengan kalimat ID (satu per baris). Gunakan tokenizer multilingual (`bert-base-multilingual-cased`) dan sesuaikan `pretrain_weight`.

---

## Referensi

- Liu et al., RoBERTa (2019) — tanpa NSP  
- [shawroad/NLP_pytorch_project](https://github.com/shawroad/NLP_pytorch_project)
