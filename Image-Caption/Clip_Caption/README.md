# Hands-On: ClipCap — CLIP + GPT-2 untuk Caption Bahasa

**Mata kuliah:** Pemrosesan Bahasa Alami Berbasis Transformer  
**Program Studi Sains Data · Fakultas Sains · Institut Teknologi Sumatera**

**Skrip utama:** `run_train_clip_gpt.py`, `inference.py`  
**Referensi:** [ClipCap](https://arxiv.org/pdf/2111.09734.pdf) · [ClipCap-Chinese](https://github.com/yangjianxin1/ClipCap-Chinese)

---

## 1. Teori dan konsep

### 1.1 ClipCap

**CLIP** (Radford et al.) memetakan gambar dan teks ke ruang embedding bersama. **ClipCap** memproyeksikan vektor gambar CLIP menjadi **prefix** — urutan embedding awal yang “mengkondisikan” **GPT-2** untuk menghasilkan caption.

```
Gambar → CLIP encoder → vektor [512]
              ↓
        MLP / BertMapper → prefix [prefix_len × d_model]
              ↓
        concat dengan embedding teks → GPT-2 → caption
```

### 1.2 Dua varian pemetaan (`model.py`)

| `mapping_type` | Modul | Keterangan |
|----------------|-------|------------|
| `MLP` | `MLP(clip_size → …)` | Proyeksi sederhana |
| `BertMapper` | BERT pada `inputs_embeds` | Prefix + konstanta trainable |

### 1.3 Pipeline data (folder `data/`)

| Skrip | Fungsi |
|-------|--------|
| `step1_run_unzip_file.py` | Ekstrak arsip Flickr |
| `step2_process_caption.py` | Bersihkan teks caption |
| `step30–32_*_encode_*.py` | Encode gambar dengan CLIP → vektor |
| `step4_concat_caption_and_vec.py` | Gabung path gambar, vektor, caption |

---

## 2. Persiapan

### Dependensi

```bash
pip install torch transformers clip-by-openai tqdm scikit-image pillow
```

### Model pra-latih (letakkan di folder modul)

| Path | Model |
|------|--------|
| `gpt2_pretrain/` | GPT-2 (tokenizer + config) |
| `bert_pretrain/` | Config BERT (jika `BertMapper`) |
| `clip_pretrain/ViT-B-32.pt` | CLIP ViT-B/32 |

### Data terproses

Setelah pipeline `data/`, dataset dibaca oleh `ClipCapDataset` di `data_helper.py` (vektor CLIP + token caption).

---

## 3. Langkah hands-on

### Langkah 1 — Pra-pemrosesan (jika mulai dari raw)

```bash
cd Image_Caption/Clip_Caption
python data/step1_run_unzip_file.py
python data/step2_process_caption.py
# ... langkah encode CLIP sesuai README asli repositori ClipCap-Chinese
python data/step4_concat_caption_and_vec.py
```

### Langkah 2 — Pelatihan

```bash
python run_train_clip_gpt.py
```

- Optimizer: AdamW + linear warmup schedule.  
- Loss: cross-entropy pada token caption (shift logits/labels).  
- Checkpoint: `output/base_model_epoch{N}_step{S}.bin`.  
- Log validasi: `output/logs.txt`.

Hyperparameter (`config.py`):

| Parameter | Default |
|-----------|---------|
| `lr` | 3e-5 |
| `prefix_len` | 10 |
| `clip_size` | 512 |
| `max_len` | 100 |
| `eval_step` | 10000 |

### Langkah 3 — Inferensi

```bash
python inference.py
```

Alur: muat GPT-2 + `ClipCaptionModel` → encode gambar dengan CLIP → **generasi autoregresif** (top-k / nucleus sampling) → decode token ke teks.

Sesuaikan `img_path` dan path checkpoint di `inference.py`.

---

## 4. Metrik evaluasi

| Metrik | Cara |
|--------|------|
| **Train / dev loss** | CE token (dicetak per step / `evaluate()`) |
| **BLEU / METEOR / CIDEr** | Evaluasi offline pada set uji (disarankan `pycocoevalcap`) |
| **Kualitatif** | 20 gambar: caption referensi vs model |

**Catatan:** modul ini fokus pada loss; untuk laporan kampus tambahkan skrip evaluasi BLEU-4 pada folder `output/`.

Contoh BLEU (setelah simpan hipotesis ke file):

```python
from nltk.translate.bleu_score import corpus_bleu
# references: list of list of token lists
# hypotheses: list of token lists
print(corpus_bleu(references, hypotheses))
```

---

## 5. Deploy

### 5.1 Serving satu gambar

```python
import clip
from PIL import Image

model, preprocess = clip.load('ViT-B-32', device='cuda')
image = preprocess(Image.open('foto.jpg')).unsqueeze(0).cuda()
with torch.no_grad():
    clip_embed = model.encode_image(image).float()
caption = generate(clip_caption_model, clip_embed, tokenizer)
```

### 5.2 API + GPU

- Simpan `ClipCaptionModel` + CLIP di memori GPU.  
- Batasi `max_len` generasi untuk latensi.  
- Half precision (`model.half()`) seperti di `inference.py` untuk throughput.

### 5.3 Multibahasa

Latih ulang pada caption Indonesia; gunakan tokenizer GPT-2 multilingual atau model bahasa setara.

---

## 6. Troubleshooting

| Masalah | Solusi |
|---------|--------|
| `gpt2_pretrain` tidak ada | Unduh dari Hugging Face ke folder lokal |
| CLIP path salah | Sesuaikan `clip_pretrain/ViT-B-32.pt` |
| Loss tidak turun | Periksa alignment vektor–caption di `step4` |
| Generasi `<unk>` dominan | Naikkan `topk`, turunkan `temperature` |

---

## Referensi

- Mokady et al., *ClipCap: CLIP Prefix for Image Captioning* (2021)  
- Radford et al., *CLIP* (2021)  
- [shawroad/NLP_pytorch_project](https://github.com/shawroad/NLP_pytorch_project)
