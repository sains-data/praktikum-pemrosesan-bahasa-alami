# Hands-On: Distilasi BERT dengan Encoder Transformer 3 Lapis

**Mata kuliah:** Pemrosesan Bahasa Alami Berbasis Transformer  
**Program Studi Sains Data · Fakultas Sains · Institut Teknologi Sumatera**

**Murid:** encoder **3 lapisan** self-attention (implementasi ringkas).  
**Guru:** BERT/RoBERTa penuh (~12 lapisan).  
**Tujuan:** murid meniru logits guru (dan skor soft) dengan arsitektur jauh lebih kecil.

---

## 1. Teori dan konsep

### 1.1 Mengapa 3 lapis?

Transformer penuh mahal secara komputasi \(O(n^2)\) per lapis. Untuk tugas klasifikasi sederhana, **murid dangkal** sering cukup jika dilatih dengan distilasi dari guru yang kuat.

### 1.2 Alur pelatihan

```
train_bert.py      →  latih / muat Model guru (BERT)
train_distill.py   →  latih Distill_Model (3-layer encoder + head)
data_process.py    →  fitur pickle (.pkl.gz)
convert_to_id.py   →  teks → id token
```

### 1.3 Loss (ringkas)

Pada `train_distill.py`, loss menggabungkan:

- Klasifikasi terhadap **label keras**  
- Kesesuaian dengan **skor/logits guru** (soft target)

---

## 2. Persiapan

```bash
pip install torch transformers sklearn tqdm
```

Siapkan `roberta_pretrain/` (vocab + config + bobot) dan fitur `.pkl.gz` dari `data_process.py`.

---

## 3. Langkah hands-on

### Langkah 1 — Data

```bash
cd Distillation/three_layer_self-attention_to_distill_bert
python data_process.py
python convert_to_id.py   # jika diperlukan pipeline Anda
```

### Langkah 2 — Model guru

```bash
python train_bert.py
```

Checkpoint contoh: `save_model/epoch{N}_ckpt.bin`

### Langkah 3 — Distilasi

```bash
python train_distill.py
```

Murid disimpan di `save_model/` (`best_pytorch_model.bin`, `epoch{N}_ckpt.bin`).

### Langkah 4 — Skor inferensi (opsional)

```bash
python inference_score.py
```

---

## 4. Metrik evaluasi

Fungsi `evaluate()` melaporkan:

| Metrik | Berkas |
|--------|--------|
| `eval_loss` | Loss gabungan pada subset eval |
| `eval_accuracy` | `accuracy_score` |
| `eval_recall` | `recall_score` |

Log ditambahkan ke `result_eval.txt`.

**Metrik tambahan untuk laporan:**

- **F1 macro** (tambahkan di skrip)  
- **Perbandingan ukuran parameter** guru vs murid  
- **Throughput** sampel/detik  

---

## 5. Deploy

```python
import torch
from Distill_Model import Model

student = Model()
student.load_state_dict(torch.load('save_model/best_pytorch_model.bin', map_location='cpu'))
student.eval()
# input_ids, input_mask, segment_ids → logits
```

**Serving:** murid 3-lapis ~3–5× lebih cepat dari BERT base pada GPU yang sama (ukur di lingkungan Anda).

---

## 6. Konfigurasi (`config.py`)

| Parameter | Default |
|-----------|---------|
| `num_train_epochs` | 10 |
| `train_batch_size` | 2 |
| `learning_rate` | 5e-5 |
| `alpha` | 0.3 (bobot komponen loss, jika dipakai) |

---

## Referensi

- [shawroad/NLP_pytorch_project](https://github.com/shawroad/NLP_pytorch_project)  
- Jiao et al., *TinyBERT* (konsep distilasi berlapis — bandingkan dengan modul `tiny_bert`)
