# Hands-On: Image Captioning — ResNet + LSTM + Attention

**Mata kuliah:** Pemrosesan Bahasa Alami Berbasis Transformer  
**Program Studi Sains Data · Fakultas Sains · Institut Teknologi Sumatera**

**Skrip utama:** `run_train.py`, `inference.py`, `server.py`  
**Referensi:** [Show, Attend and Tell](https://arxiv.org/pdf/1502.03044.pdf) · [foamliu/Image-Captioning-PyTorch](https://github.com/foamliu/Image-Captioning-PyTorch)

---

## 1. Teori dan konsep

### 1.1 Image captioning

Tugas: \( f: \text{gambar} \rightarrow \text{urutan kata} \). Model harus memahami objek/adegan visual lalu menghasilkan kalimat gramatikal.

### 1.2 Arsitektur Show, Attend and Tell

```
Gambar  →  Encoder (ResNet-101)  →  fitur spasial 14×14×2048
                                              ↓
                         Attention (soft alignment per langkah)
                                              ↓
                         Decoder (LSTM)  →  token berikutnya
```

- **Encoder:** ResNet-101 pra-latih, fine-tune lapisan atas.  
- **Attention:** Bobot \(\alpha_{t,i}\) pada setiap piksel/feature map saat menghasilkan kata ke-\(t\).  
- **Decoder:** LSTM memprediksi kata; loss cross-entropy + regularisasi attention ganda stokastik.

### 1.3 Berkas penting

| Berkas | Peran |
|--------|--------|
| `model.py` | `Encoder`, `Attention`, `DecoderWithAttention` |
| `data_helper.py` | Dataset caption (5 caption/gambar) |
| `data/gen_vocab.py` | Bangun `WORDMAP.json` |
| `run_train.py` | Latih + validasi BLEU-4 |
| `inference.py` | Beam search pada satu gambar |
| `server.py` | Demo web Streamlit |

---

## 2. Persiapan data

Struktur umum (Flickr-style):

```
data/
  flickr8k_images/     # atau flickr30k-images/
  dataset_flickr8k.json   # image_id + captions
  WORDMAP.json         # dari gen_vocab.py
```

Format JSON: setiap gambar punya beberapa deskripsi (default 5 per gambar di `CaptionDataset`).

```bash
cd Image_Caption/resnet_rnn
python data/gen_vocab.py   # setelah JSON siap
```

---

## 3. Langkah hands-on

### Langkah 1 — Konfigurasi

Edit `config.py` atau argumen default:

| Parameter | Default | Keterangan |
|-----------|---------|------------|
| `epochs` | 200 | Epoch pelatihan |
| `batch_size` | 64 | |
| `encoder_lr` / `decoder_lr` | 4e-4 / 1e-4 | LR terpisah |
| `alpha_c` | 1.0 | Bobot regularisasi attention |
| `max_len` | 40 | Panjang caption |

### Langkah 2 — Pelatihan

```bash
pip install torch torchvision nltk scipy pillow
python run_train.py
```

Checkpoint disimpan via `utils.save_checkpoint`. Validasi mencetak **loss**, **top-5 accuracy**, dan **BLEU-4**.

### Langkah 3 — Inferensi

```bash
python inference.py
```

Muat encoder/decoder + `WORDMAP.json`, jalankan **beam search** (`beam_size=3`).

### Langkah 4 — Deploy demo (Streamlit)

```bash
pip install streamlit opencv-python
streamlit run server.py
```

Unggah gambar JPG → tampilkan caption yang dihasilkan.

Alternatif background:

```bash
bash start.sh   # nohup pelatihan di GPU 1
```

---

## 4. Metrik evaluasi

| Metrik | Arti | Lokasi |
|--------|------|--------|
| **Training / val loss** | CE pada kata target + reg. attention | `run_train.py` |
| **Top-5 accuracy** | Akurasi kata dalam top-5 | `evaluate()` |
| **BLEU-4** | Kesamaan n-gram hipotesis vs referensi | `corpus_bleu` (NLTK) |
| **METEOR / CIDEr** (opsional) | Tambahkan via `pycocoevalcap` untuk laporan |

**Eksperimen laporan:**

1. BLEU-4 vs epoch.  
2. Contoh 5 gambar: referensi vs prediksi.  
3. Visualisasi peta attention (`alphas` dari decoder).  
4. Bandingkan `beam_size` 1 vs 3.

---

## 5. Deploy produksi

### 5.1 API Flask/FastAPI

```python
# Muat encoder, decoder, word_map sekali saat startup
caption = caption_image_beam_search(encoder, decoder, path, word_map)
```

### 5.2 Optimasi inferensi

- Cache `encoder_out` per gambar jika banyak query.  
- TorchScript / ONNX untuk ResNet encoder.  
- Batch inference untuk katalog gambar.

### 5.3 Bahasa

Kosakata contoh (`WORDMAP.json`) berbahasa Mandarin. Untuk Indonesia: latih ulang pada Flickr-ID atau dataset lokal + rebuild vocab.

---

## 6. Troubleshooting

| Masalah | Solusi |
|---------|--------|
| `pretrained=True` deprecated | Ganti `weights=ResNet101_Weights.IMAGENET1K_V1` di PyTorch baru |
| `scipy.misc.imread` error | Gunakan `imageio` / `PIL` (versi SciPy lama) |
| BLEU sangat rendah | Periksa path gambar, vocab, epoch cukup |
| OOM | Kurangi `batch_size` |

---

## Referensi

- Xu et al., *Show, Attend and Tell* (2015)  
- [shawroad/NLP_pytorch_project](https://github.com/shawroad/NLP_pytorch_project)
