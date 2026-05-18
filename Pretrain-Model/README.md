# Pretrain Model — Pra-Pelatihan Model Bahasa

**Mata kuliah:** Pemrosesan Bahasa Alami Berbasis Transformer  
**Program Studi Sains Data · Fakultas Sains · Institut Teknologi Sumatera**

Modul ini mendemonstrasikan **continued pre-training** pada korpus domain sendiri dengan tiga varian:

| Varian | Folder | Fokus tugas |
|--------|--------|-------------|
| BERT (MLM + NSP) | [`bert-pretrain/`](bert-pretrain/) | Masked LM + next sentence prediction |
| Mask-only (RoBERTa) | [`mask_pretrain/`](mask_pretrain/) | Hanya MLM / mask (tanpa NSP) |
| WoBERT | [`wobert-pretrain/`](wobert-pretrain/) | MLM dengan whole-word masking Mandarin |

| Panduan | Tautan |
|---------|--------|
| BERT | [bert-pretrain/README.md](bert-pretrain/README.md) |
| Mask-only | [mask_pretrain/README.md](mask_pretrain/README.md) |
| WoBERT | [wobert-pretrain/README.md](wobert-pretrain/README.md) |

**Sumber asli:** [shawroad/NLP_pytorch_project](https://github.com/shawroad/NLP_pytorch_project) — folder `Pretrain Model`.

---

## Kapan memakai pra-pelatihan?

- Domain khusus (hukum, medis, bahasa daerah) dengan sedikit data berlabel  
- Ingin meningkatkan representasi sebelum fine-tune klasifikasi / NER / QA  
- Eksperimen kurikulum: BERT-base → continued pretrain → task-specific fine-tune

---

## Perbandingan singkat

| | bert-pretrain | mask_pretrain | wobert-pretrain |
|---|---------------|---------------|-----------------|
| Tokenizer | BERT-base CJK | RoBERTa/BERT | WoBERT (whole word) |
| NSP | Ya | Tidak | Ya (via BertForPreTraining) |
| Data prep | `corpus/` + `get_train_data.py` | Satu kalimat per baris | `process_pretrain_data.py` |
