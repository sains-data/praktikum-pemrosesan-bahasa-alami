# Hands-On: TinyBERT — Distilasi General + MSE Antar Lapisan

**Mata kuliah:** Pemrosesan Bahasa Alami Berbasis Transformer  
**Program Studi Sains Data · Fakultas Sains · Institut Teknologi Sumatera**

Implementasi mengikuti ide **TinyBERT** (Huawei): selain mencocokkan **logits** guru dan murid, murid juga meniru **hidden/attention** pada lapisan tertentu (MSE).

---

## 1. Teori dan konsep

### 1.1 Dua tahap TinyBERT (paper)

1. **General distillation** — murid meniru guru pada data domain umum (ditunjukkan di sini dengan klasifikasi + matching lapisan).  
2. **Task-specific distillation** — fine-tune pada tugas downstream (bisa dilanjutkan setelah modul ini).

### 1.2 Loss di `train_distill_v2.py`

\[
\mathcal{L} = \mathcal{L}_{CE} + \mathcal{L}_{MSE}^{logit} + \sum_{i} \mathcal{L}_{MSE}^{attn}(h_s^{(i)}, h_t^{(j)})
\]

Mapping lapisan (contoh di kode):

| Murid (3 lapis) | Guru (BERT 12 lapis) |
|-----------------|----------------------|
| Lapis 1 | Lapis 1 |
| Lapis 2 | Lapis 6 |
| Lapis 3 | Lapis 12 |

Fungsi: `compute_loss()` — `loss1` (CE), `loss2` (kd MSE logits), `layer*_mse_loss` (attention).

### 1.3 Arsitektur berkas

| Berkas | Peran |
|--------|--------|
| `model.py` | BERT guru (`Model`) |
| `Distill_Model.py` | Murid `CModel` (3 encoder) |
| `train.py` | Latih guru |
| `train_distill_v2.py` | Distilasi TinyBERT |
| `data_process.py` | Pra-pemrosesan (format QA/klasifikasi) |

---

## 2. Langkah hands-on

### Langkah 1 — Pra-pemrosesan

```bash
cd Distillation/tiny_bert
python data_process.py
```

Menghasilkan `train_features.pkl.gz` (atau path di `config.py`).

### Langkah 2 — Latih guru

```bash
python train.py
```

Checkpoint: `save_model/epoch{N}_ckpt.bin`

### Langkah 3 — Distilasi TinyBERT

```bash
python train_distill_v2.py
```

Pastikan `teacher_model.load_state_dict` menunjuk ke checkpoint guru yang benar (`epoch3_ckpt.bin` di kode asli).

Murid terbaik: `save_model/best_pytorch_model.bin`

---

## 3. Metrik evaluasi

| Metrik | Sumber |
|--------|--------|
| `eval_loss` | Loss gabungan distilasi |
| `eval_accuracy` | Akurasi klasifikasi |
| `eval_recall` | Recall (binary/multiclass) |

Disimpan di `result_eval.txt` tiap evaluasi.

**Eksperimen:**

1. Murid **tanpa** MSE lapisan vs **dengan** (`train_distill_v2.py`).  
2. Akurasi vs jumlah parameter.  
3. Waktu latih & inferensi guru vs murid.

---

## 4. Deploy

```python
import torch
from Distill_Model import CModel

student = CModel(device='cpu')
student.load_state_dict(torch.load('save_model/best_pytorch_model.bin', map_location='cpu'))
student.eval()
```

**Produksi:**

- Gunakan murid untuk inferensi real-time; guru hanya untuk re-distilasi berkala.  
- Suhu distilasi (`temperature` di `kd_mse_loss`) dapat dituning (default 1).

---

## 5. Hyperparameter (`config.py`)

| Parameter | Default |
|-----------|---------|
| `alpha` | 0.3 |
| `learning_rate` | 5e-5 |
| `num_train_epochs` | 10 |
| `train_batch_size` | 2 |
| `gradient_accumulation_steps` | 1 |

---

## 6. Troubleshooting

| Masalah | Solusi |
|---------|--------|
| `train_features.pkl.gz` tidak ada | `data_process.py` |
| OOM | Kurangi `batch_size`, gradient accumulation |
| Loss NaN | Clip gradient; kurangi LR |
| Teacher path salah | Sesuaikan `load_state_dict` di `train_distill_v2.py` |

---

## Referensi

- Jiao et al., *TinyBERT: Distilling BERT for Natural Language Understanding* (2019)  
- [shawroad/NLP_pytorch_project](https://github.com/shawroad/NLP_pytorch_project)
