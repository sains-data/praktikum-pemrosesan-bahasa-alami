# NER — Named Entity Recognition

**Mata kuliah:** Pemrosesan Bahasa Alami Berbasis Transformer  
**Program Studi Sains Data · Fakultas Sains · Institut Teknologi Sumatera**

Modul ini membandingkan empat pendekatan **Named Entity Recognition (NER)** — pengenalan entitas bernama dalam teks.

| Pendekatan | Folder | Lapisan decode |
|------------|--------|----------------|
| BiLSTM + CRF | [`BiLSTM_CRF_Ner/`](BiLSTM_CRF_Ner/) | Klasik, tanpa transformer |
| BERT + Softmax | [`Bert_Softmax_Ner/`](Bert_Softmax_Ner/) | Label per token (BIO) |
| BERT + CRF | [`Bert_CRF_Ner/`](Bert_CRF_Ner/) | CRF di atas representasi BERT |
| GlobalPointer | [`GlobalPointer/`](GlobalPointer/) | Span-based (RoBERTa + pointer) |

| Panduan hands-on | Tautan |
|------------------|--------|
| BiLSTM-CRF | [bilstm_crf/README.md](bilstm_crf/README.md) |
| BERT-Softmax | [bert_softmax/README.md](bert_softmax/README.md) |
| BERT-CRF | [bert_crf/README.md](bert_crf/README.md) |
| GlobalPointer | [global_pointer/README.md](global_pointer/README.md) |

**Sumber asli:** [shawroad/NLP_pytorch_project](https://github.com/shawroad/NLP_pytorch_project) — folder `NER`.

---

## Format data umum

| Format | Contoh modul | Contoh baris |
|--------|--------------|--------------|
| **BMES/BIO per karakter** | BERT-CRF, BERT-Softmax | `吴 B-NAME` lalu baris kosong antar kalimat |
| **JSON span** | GlobalPointer | `{"text": "...", "label": {"name": [[0,2], ...]}}` |
| **Renmin (kata/tag)** | BiLSTM-CRF | `中/B-ns` dari korpus People Daily |

**Skema label MSRA (contoh):** `O`, `B-NAME`, `M-NAME`, `E-NAME`, `B-ORG`, … (format BMES).

---

## Alur belajar disarankan

1. **BiLSTM-CRF** — pahami CRF dan tagging sekuens.  
2. **BERT-Softmax** — encoder pra-latih + klasifikasi token.  
3. **BERT-CRF** — constraint transisi label.  
4. **GlobalPointer** — NER span tanpa decoding Viterbi per token.
