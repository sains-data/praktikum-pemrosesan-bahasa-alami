# Image Captioning (Keterangan Gambar)

**Mata kuliah:** Pemrosesan Bahasa Alami Berbasis Transformer  
**Program Studi Sains Data · Fakultas Sains · Institut Teknologi Sumatera**

Modul ini mempelajari tugas **image captioning**: menghasilkan deskripsi teks alami dari gambar. Dua pendekatan disediakan: klasik **CNN + RNN + attention** dan modern **CLIP + GPT-2**.

| Modul | Arsitektur | Panduan |
|-------|------------|---------|
| `resnet_rnn` | ResNet-101 + LSTM + attention (Show, Attend and Tell) | [resnet_rnn/README.md](resnet_rnn/README.md) |
| `Clip_Caption` | CLIP embedding + prefix + GPT-2 (ClipCap) | [Clip_Caption/README.md](Clip_Caption/README.md) |

**Sumber asli:** [shawroad/NLP_pytorch_project](https://github.com/shawroad/NLP_pytorch_project) — folder `Image_Caption`.

## Perbandingan singkat

| Aspek | ResNet + RNN | CLIP + GPT-2 |
|-------|----------------|--------------|
| Encoder visual | ResNet pra-latih ImageNet | CLIP ViT |
| Generator teks | LSTM + attention | GPT-2 autoregresif |
| Data tipikal | Flickr8k / Flickr30k (JSON + gambar) | Flickr + embedding CLIP |
| Deploy demo | `server.py` (Streamlit) | `inference.py` |
