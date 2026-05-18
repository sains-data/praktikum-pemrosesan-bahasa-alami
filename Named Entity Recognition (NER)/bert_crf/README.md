# Hands-On: BERT + CRF untuk NER

**Skrip utama:** [`../Bert_CRF_Ner/run_ner_crf.py`](../Bert_CRF_Ner/run_ner_crf.py)  
**Inferensi:** [`../Bert_CRF_Ner/inference.py`](../Bert_CRF_Ner/inference.py)

---

## 1. Teori

**BERT** menghasilkan embedding kontekstual per token/subword. **CRF** memodelkan dependensi antar label (mis. `B-NAME` tidak boleh langsung diikuti `E-ORG`), sehingga urutan tag lebih konsisten daripada Softmax independen.

```
Teks → BERT → Linear → CRF → urutan tag BMES
```

---

## 2. Data

- `data/train.char.bmes`, `dev.char.bmes`, `test.char.bmes`  
- Satu karakter per baris + label; baris kosong = akhir kalimat  

Unduh `bert_config.json` dan `pytorch_model.bin` ke `bert_pretrain/` (lihat catatan di folder tersebut).

---

## 3. Langkah hands-on

```bash
cd NER/Bert_CRF_Ner
pip install torch transformers
python run_ner_crf.py
```

Uji satu kalimat:

```bash
python inference.py
```

Hyperparameter: `config.py` → `get_argparse()` (`max_seq_length`, `learning_rate`, `epochs`).

---

## 4. Metrik

Modul `metrics.py` — **SeqEntityScore**: precision, recall, F1 per entitas dan rata-rata (span-level setelah decode BMES).

| Metrik | Arti |
|--------|------|
| **Entity F1** | Span entitas tepat (tipe + batas) |
| **Precision / Recall** | Per kelas NAME, ORG, TITLE, … |

---

## 5. Deploy

```python
# Muat checkpoint + tokenizer, jalankan inference.py sebagai template
entities = get_entities(pred_tags, id2label)
```

- **API:** `POST /ner` → JSON daftar `{"text": "...", "type": "NAME", "start": 0, "end": 2}`  
- **Batch:** proses dokumen hukum / resume (data contoh profil eksekutif Mandarin)

---

## Referensi

- [shawroad/NLP_pytorch_project](https://github.com/shawroad/NLP_pytorch_project)
