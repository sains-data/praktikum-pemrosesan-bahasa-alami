# Hands-On: PEFT pada Klasifikasi Sentimen (BERT)

**Mata kuliah:** Pemrosesan Bahasa Alami Berbasis Transformer · Itera  
**Folder kerja:** `peft_example/sequence_cls/bert/`

---

## 1. Teori

### 1.1 PEFT (Parameter-Efficient Fine-Tuning)

Melatih seluruh bobot LLM/BERT besar mahal. **PEFT** hanya melatih subset kecil parameter:

| Metode | Berkas latih | Berkas inferensi |
|--------|--------------|------------------|
| **LoRA** | `run_train_bert_lora.py` | `run_infer_bert_lora.py` |
| **P-Tuning** | `run_train_bert_p_tuning.py` | `run_infer_bert_p_tuning.py` |
| **Prefix Tuning** | `run_train_bert_prefix_tuning.py` | `run_infer_bert_prefix_tuning.py` |
| **Prompt Tuning** | `run_train_bert_prompt_tuning.py` | `run_infer_bert_prompt_tuning.py` |

LoRA menambahkan matriks rank-rendah pada layer attention; P-Tuning/Prefix/Prompt menambahkan vektor virtual di input.

### 1.2 Data

- `data/train.csv`, `data/val.csv` — teks Weibo + label biner (0/1)  
- Tokenizer & config: `mengzi_pretrain/` (BERT Mandarin; bisa diganti checkpoint Indonesia)

---

## 2. Instalasi

```bash
pip install torch transformers peft pandas scikit-learn tqdm
```

---

## 3. Langkah hands-on

```bash
cd LLM/peft_example/sequence_cls/bert
python run_train_bert_lora.py
```

Uji inferensi setelah checkpoint tersimpan di `./lora/`:

```bash
python run_infer_bert_lora.py
```

Bandingkan metode lain dengan skrip `run_train_bert_*` dan `run_infer_bert_*` yang sesuai.

**Konfigurasi:** [`config.py`](config.py) — path data, epoch, batch size, learning rate.

**Kustom LoRA pada modul tertentu:** lihat komentar di [`run_customized_by_oneself_lora.py`](run_customized_by_oneself_lora.py).

---

## 4. Metrik evaluasi

| Metrik | Implementasi |
|--------|----------------|
| **Accuracy** | `sklearn.metrics.accuracy_score` di fungsi `evaluate()` |
| **F1 / precision / recall** | Tambahkan `classification_report` pada val set |
| **Trainable %** | `model.print_trainable_parameters()` (~0,3% untuk LoRA contoh) |

**Checklist laporan:**

- [ ] Tabel perbandingan LoRA vs P-Tuning vs full fine-tune (jika ada)  
- [ ] Waktu latih & ukuran checkpoint  
- [ ] Contoh prediksi 5 tweet

---

## 5. Deploy

```python
from peft import PeftModel
from transformers import BertForSequenceClassification, BertTokenizer

base = BertForSequenceClassification.from_pretrained("./mengzi_pretrain", num_labels=2)
model = PeftModel.from_pretrained(base, "./lora")
tokenizer = BertTokenizer.from_pretrained("./mengzi_pretrain")
# tokenize → model(**inputs) → argmax logits
```

- **Batch:** skor sentimen harian pada feed media sosial  
- **API:** FastAPI `POST /sentiment` dengan body `{"text": "..."}`

---

## 6. Catatan data Indonesia

Ganti `train.csv` / `val.csv` dengan kolom `text,label` (0 negatif, 1 positif). Sesuaikan `pretrained_model_path` ke model multilingual, mis. `indobenchmark/indobert-base-p1`.

---

## Referensi

- [PEFT library](https://github.com/huggingface/peft)  
- Hu et al., LoRA (2021)  
- [shawroad/NLP_pytorch_project](https://github.com/shawroad/NLP_pytorch_project)
