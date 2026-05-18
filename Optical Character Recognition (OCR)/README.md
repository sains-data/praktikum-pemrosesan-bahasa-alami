# OCR — Optical Character Recognition

**Mata kuliah:** Pemrosesan Bahasa Alami Berbasis Transformer  
**Program Studi Sains Data · Fakultas Sains · Institut Teknologi Sumatera**

Modul ini mendemonstrasikan **OCR** untuk mengekstrak teks dari gambar, dengan contoh pipeline **subtitle video** (deteksi area bawah layar + pengenalan teks).

| Komponen | Lokasi |
|----------|--------|
| Ekstraksi subtitle dari video | [`ekstraksi_subtitle_video/`](ekstraksi_subtitle_video/) |
| Panduan hands-on | [ekstraksi_subtitle_video/README.md](ekstraksi_subtitle_video/README.md) |

**Sumber asli:** [shawroad/NLP_pytorch_project](https://github.com/shawroad/NLP_pytorch_project) — folder `OCR` (subfolder asli berbahasa Mandarin telah diganti nama).

---

## Ringkasan

Demo ini memakai model pra-latih **PaddleHub** (`chinese_ocr_db_crnn_server`) — deteksi kotak teks (DB) + pengenalan (CRNN), tanpa melatih dari nol.

**Alur dua langkah:**

```
video.mp4 → step1 (1 frame/detik, crop 1/4 bawah) → frame/*.jpg
         → step2 (OCR batch) → daftar teks subtitle
```

---

## Prasyarat umum

- Python 3.8+
- OpenCV, PaddlePaddle, PaddleHub (lihat panduan subfolder)
- File video contoh: `ekstraksi_subtitle_video/data/video.mp4`
