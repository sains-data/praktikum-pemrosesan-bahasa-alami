# Hands-On: DynaBERT — Pruning & Distilasi Struktur BERT

**Mata kuliah:** Pemrosesan Bahasa Alami Berbasis Transformer  
**Program Studi Sains Data · Fakultas Sains · Institut Teknologi Sumatera**

Implementasi terinspirasi **DynaBERT** (Huawei): melatih model guru BERT penuh, lalu **memangkas** (prune) jumlah lapisan, kepala attention, dan dimensi tersembunyi untuk mendapatkan varian lebih kecil dan cepat.

---

## 1. Teori dan konsep

### 1.1 Knowledge distillation vs pruning

| Teknik | Ide |
|--------|-----|
| **Distilasi** | Murid meniru keluaran guru (logit, attention) |
| **Pruning** | Menghapus unit/lapis yang kurang penting |
| **DynaBERT** | Gabungan: satu guru, banyak konfigurasi murid (elastic) via pruning bertahap |

### 1.2 Arsitektur alur proyek

```
train_teacher_model.py  →  BERT guru (Teacher_Model)
         │
         ▼
train_tailor_model.py   →  BERT dipangkas + distilasi dari guru
train_tailor_model_v2.py →  varian alternatif tailoring
```

**Berkas penting:**

| Berkas | Peran |
|--------|--------|
| `model_teacher.py` | Arsitektur BERT guru |
| `config_teacher.py` / `config_tailor.py` | Hyperparameter |
| `data_process.py` | Fitur ke `train_features.pkl.gz` |
| `my_transformers/` | Fork HuggingFace BERT (versi lama) |

---

## 2. Persiapan

```bash
pip install torch transformers sklearn tqdm
```

Siapkan:

- `bert_pretrain/` — `pytorch_model.bin`, `bert_config.json`, `vocab.txt`
- `data/train_features.pkl.gz`, `data/dev_features.pkl.gz` (dari `data_process.py`)

---

## 3. Langkah hands-on

### Langkah 1 — Pra-pemrosesan

```bash
cd Distillation/DynaBert
python data_process.py
```

### Langkah 2 — Latih model guru

```bash
python train_teacher_model.py
```

Checkpoint disimpan di `save_teacher_model/` (lihat `config_teacher.py`).

### Langkah 3 — Pruning / tailor model murid

```bash
python train_tailor_model.py
# atau
python train_tailor_model_v2.py
```

Model murid belajar dengan **soft cross-entropy** terhadap logits guru sambil struktur BERT dikurangi.

---

## 4. Metrik evaluasi

| Metrik | Keterangan |
|--------|------------|
| **Accuracy** | `sklearn.metrics.accuracy_score` pada set dev |
| **Recall / Precision** | Dicetak di `evaluate()` — `train_teacher_model.py` |
| **Eval loss** | Cross-entropy pada dev |
| **Ukuran model** | Jumlah parameter sebelum/sesudah prune |
| **Latency** | Waktu inferensi per batch (ukur manual) |

**Eksperimen laporan:**

1. Akurasi guru vs murid ter-prune.  
2. Rasio kecepatan inferensi (ms/sampel).  
3. Trade-off akurasi vs % parameter dihapus.

Hasil evaluasi juga ditulis ke `result_eval.txt`.

---

## 5. Deploy

### 5.1 Inferensi murid

```python
import torch
from model_teacher import Teacher_Model  # atau model tailor setelah prune

model = Teacher_Model(...)
model.load_state_dict(torch.load('save_teacher_model/best.bin', map_location='cpu'))
model.eval()
# forward: input_ids, input_mask, segment_ids
```

### 5.2 Produksi

- Deploy varian **paling kecil** yang masih memenuhi SLA akurasi.  
- Simpan konfigurasi prune (jumlah layer, heads) di metadata model.  
- DynaBERT cocok untuk **serving multi-SLA** (model besar untuk akurasi, kecil untuk latensi).

### 5.3 ONNX / TorchScript

Setelah prune stabil, ekspor murid ke ONNX untuk serving CPU.

---

## 6. Troubleshooting

| Masalah | Solusi |
|---------|--------|
| `.pkl.gz` tidak ada | Jalankan `data_process.py` |
| CUDA error | Set `n_gpu` dan device di config |
| Akurasi murid jatuh drastis | Kurangi agresivitas prune; latih tailor lebih lama |

---

## Referensi

- Hou et al., *DynaBERT: Dynamic BERT with Adaptive Width and Depth* (2020)  
- [shawroad/NLP_pytorch_project](https://github.com/shawroad/NLP_pytorch_project)
