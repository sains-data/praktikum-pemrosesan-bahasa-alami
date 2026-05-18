# Hands-On: Hugging Face Accelerate

**Mata kuliah:** Pemrosesan Bahasa Alami Berbasis Transformer · Itera  
**Skrip:** [`nlp_example.py`](nlp_example.py)

---

## 1. Teori

**Accelerate** adalah library Hugging Face yang menyatukan pelatihan di:

- Satu GPU, multi-GPU (DDP), TPU, CPU  
- Mixed precision (`fp16`, `bf16`, `fp8`)  
- Gradient accumulation otomatis  

Tanpa menulis loop DDP manual, Anda memanggil `accelerator.prepare(model, optimizer, dataloader)` dan `accelerator.backward(loss)`.

Contoh ini melatih **BERT-base-cased** pada tugas **GLUE MRPC** (paraphrase detection).

---

## 2. Instalasi

```bash
pip install accelerate torch transformers datasets evaluate
```

---

## 3. Langkah hands-on

```bash
cd LLM/accelerate_example
python nlp_example.py
```

Multi-GPU (contoh 2 kartu):

```bash
accelerate launch nlp_example.py
```

Opsi mixed precision:

```bash
python nlp_example.py --mixed_precision fp16
```

---

## 4. Metrik evaluasi

Skrip memuat metrik GLUE MRPC via `evaluate.load("glue", "mrpc")`:

| Metrik | Arti |
|--------|------|
| **accuracy** | Proporsi prediksi benar |
| **f1** | F1 untuk kelas positif (paraphrase) |

Output per epoch dicetak oleh `accelerator.print` (hanya proses utama).

---

## 5. Deploy

Accelerate dipakai di **pipeline pelatihan**, bukan serving. Setelah model terbaik disimpan:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model = AutoModelForSequenceClassification.from_pretrained("./checkpoint")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
```

Serving: FastAPI + model ONNX, atau `pipeline("text-classification")`.

---

## Referensi

- [Accelerate docs](https://huggingface.co/docs/accelerate)  
- [shawroad/NLP_pytorch_project](https://github.com/shawroad/NLP_pytorch_project)
