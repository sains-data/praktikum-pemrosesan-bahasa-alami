# Distilasi Model (Knowledge Distillation) untuk NLP

**Mata kuliah:** Pemrosesan Bahasa Alami Berbasis Transformer  
**Program Studi Sains Data · Fakultas Sains · Institut Teknologi Sumatera**

Folder ini mempelajari cara **menyusutkan** model besar (guru, mis. BERT) menjadi model kecil (murid) yang lebih cepat di inferensi, dengan menjaga akurasi sebisa mungkin melalui **knowledge distillation**.

| Modul | Pendekatan | Panduan |
|-------|------------|---------|
| `DynaBert` | Pruning struktur BERT (Huawei) | [DynaBert/README.md](DynaBert/README.md) |
| `rnn_distill_bert` | BiLSTM murid + soft label dari BERT | [rnn_distill_bert/README.md](rnn_distill_bert/README.md) |
| `three_layer_self-attention_to_distill_bert` | Encoder Transformer 3 lapis → distilasi BERT | [three_layer_self-attention_to_distill_bert/README.md](three_layer_self-attention_to_distill_bert/README.md) |
| `tiny_bert` | TinyBERT: soft label + MSE antar lapisan | [tiny_bert/README.md](tiny_bert/README.md) |

**Sumber asli:** [shawroad/NLP_pytorch_project](https://github.com/shawroad/NLP_pytorch_project) — folder `Distillation`.

## Konsep umum

```
Model guru (BERT besar, akurat)  ──►  sinyal lunak (logits, perhatian, hidden)
                                           │
                                           ▼
Model murid (kecil, cepat)       ◄──  loss gabungan: hard label + distilasi
```

**Hard label:** label asli (cross-entropy).  
**Soft label:** distribusi probabilitas guru (MSE / KL / matching lapisan).
