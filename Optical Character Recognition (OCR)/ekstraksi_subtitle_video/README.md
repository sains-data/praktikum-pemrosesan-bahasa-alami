# Hands-On: OCR untuk Subtitle Video

**Langkah 1:** [`step1_extract_frame.py`](step1_extract_frame.py) — ekstraksi frame  
**Langkah 2:** [`step2_ocr_recognize.py`](step2_ocr_recognize.py) — pengenalan teks

---

## 1. Teori dan konsep

### 1.1 Apa itu OCR?

**Optical Character Recognition (OCR)** mengubah piksel teks dalam gambar menjadi string. Pipeline modern umumnya:

1. **Deteksi teks** — menemukan kotak (bounding box) di mana teks berada.  
2. **Pengenalan** — membaca karakter di dalam setiap kotak.

Model PaddleHub `chinese_ocr_db_crnn_server` menggabungkan **DB (Differentiable Binarization)** + **CRNN**.

### 1.2 Mengapa crop 1/4 bawah video?

Subtitle film/serial biasanya berada di **bagian bawah layar**. Memotong `(h//4)*3:h` mengurangi noise dari adegan dan mempercepat OCR.

### 1.3 Sampling 1 frame per detik

Mengambil setiap `fps` frame ≈ **satu gambar per detik** — cukup untuk subtitle yang berubah perlahan, dengan biaya komputasi lebih rendah.

---

## 2. Instalasi

```bash
pip install opencv-python
pip install paddlepaddle==2.2.2
pip install paddlehub==2.0.0
pip install shapely==1.8.1.post1
pip install pyclipper==1.3.0.post2
```

**GPU (opsional):** instal `paddlepaddle-gpu` yang sesuai CUDA; set `use_gpu=True` di `step2_ocr_recognize.py`.

Siapkan video:

```bash
mkdir -p data
# salin video Anda ke data/video.mp4
```

---

## 3. Langkah hands-on

```bash
cd OCR/ekstraksi_subtitle_video
mkdir -p frame

python step1_extract_frame.py
python step2_ocr_recognize.py
```

**Keluaran:**

- `frame/image_001.jpg`, … — cuplikan bawah layar per detik  
- `ocr_result/` — hasil visualisasi (jika `visualization=True`)  
- Konsol: daftar teks + gabungan unik dipisah koma

---

## 4. Metrik evaluasi

| Metrik | Arti | Cara ukur |
|--------|------|-----------|
| **Detection precision/recall** | Kotak teks terdeteksi benar | Bandingkan dengan anotasi manual pada subset frame |
| **Character / word accuracy** | Karakter/kata benar | Edit distance (CER/WER) vs ground truth subtitle |
| **Confidence** | Skor DB + CRNN | Naikkan `box_thresh`, `text_thresh` jika banyak false positive |
| **Latency** | Waktu per frame | Dicetak di akhir `step2` |

**Checklist laporan:**

- [ ] 3 contoh frame + teks OCR  
- [ ] Pengaruh `box_thresh` / `text_thresh`  
- [ ] Perbandingan crop bawah vs frame penuh  

---

## 5. Deploy

| Skenario | Pendekatan |
|----------|------------|
| **Batch subtitle** | Pipeline step1+2 pada file video; simpan SRT dengan timestamp per detik |
| **API** | FastAPI: upload frame/gambar → PaddleHub OCR → JSON `{"boxes": [...], "text": "..."}` |
| **Produksi** | Pertimbangkan PaddleOCR, EasyOCR, atau layanan cloud; tambah post-processing (deduplikasi, alignment temporal) |

**Contoh konversi ke SRT (konsep):**

```text
1
00:00:01,000 --> 00:00:02,000
{teks_dari_frame_1}
```

---

## 6. Troubleshooting

| Masalah | Solusi |
|---------|--------|
| `video.mp4` tidak ditemukan | Letakkan di `data/video.mp4` atau ubah `video_path` |
| Folder `frame` kosong | Pastikan codec video didukung OpenCV |
| OCR kosong | Turunkan threshold; perbesar area crop; cek resolusi frame |
| PaddleHub gagal instal | Sesuaikan versi Python; gunakan virtualenv |
| Teks duplikat banyak | Sudah ada deduplikasi di step2; tambah filter similarity antar frame |

---

## 7. Perluasan (Bahasa Indonesia)

- Ganti modul ke model multilingual PaddleOCR jika subtitle ID/EN.  
- Gunakan **EasyOCR** (`['id','en']`) untuk eksperimen cepat tanpa Paddle.  
- Integrasi dengan **Whisper** jika audio tersedia (ASR sering lebih akurat untuk subtitle).

---

## Referensi

- [PaddleHub OCR](https://github.com/PaddlePaddle/PaddleHub)  
- [shawroad/NLP_pytorch_project](https://github.com/shawroad/NLP_pytorch_project)
