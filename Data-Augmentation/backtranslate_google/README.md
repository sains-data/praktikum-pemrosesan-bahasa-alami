# Hands-On: Augmentasi Back-Translation (Google Translate)

**Mata kuliah:** Pemrosesan Bahasa Alami Berbasis Transformer  
**Program Studi Sains Data · Fakultas Sains · Institut Teknologi Sumatera**

**Skrip:** [`../003-google_trans_data_aug.py`](../003-google_trans_data_aug.py)

---

## 1. Teori dan konsep

### 1.1 Back-translation skala batch

Sama seperti modul Baidu, tetapi skrip ini mengoptimalkan **throughput**: menggabungkan banyak kalimat dengan pemisah `\n`, menerjemahkan sekali, lalu memecah kembali. Cocok untuk puluhan ribu kalimat pendek.

```
Kalimat 1     ┐
Kalimat 2     ├─► gabung (\n) ─► translate ─► pisah ─► daftar augmentasi
...           ┘
```

### 1.2 Token `tk` (Google)

Google Translate web memerlukan parameter `tk` yang dihitung dari teks (fungsi JavaScript `TL` di kelas `Kaihua` via `PyExecJS`). Ini **bukan API resmi** — endpoint dapat berubah atau diblokir.

### 1.3 Batasan

- ~**5000 karakter** per permintaan gabungan (komentar asli: 100 kalimat × ~50 karakter).  
- Sesuaikan `combined_length` jika kalimat panjang.  
- Domain `translate.google.cn` mungkin tidak tersedia di semua jaringan.

---

## 2. Arsitektur kode

```
003-google_trans_data_aug.py
├── open_url(url)              → HTTP GET dengan User-Agent
├── translate(content, tk)     → URL Google + parse respons
├── class Kaihua               → execjs: hitung tk dari teks
└── __main__
      ├── Gabung a_list per combined_length
      ├── Loop retry sampai jumlah baris cocok
      └── a_trans_list (hasil terjemahan)
```

Alur back-translation penuh (praktikum): jalankan dua kali dengan `sl`/`tl` berbeda (Zh→En lalu En→Zh), atau kombinasikan dengan pipeline Baidu.

---

## 3. Persiapan

```bash
pip install pandas tqdm PyExecJS
# Node.js diperlukan untuk execjs (backend default)
```

Uji koneksi ke layanan Google Translate dari jaringan kampus.

---

## 4. Langkah hands-on

### Langkah 1 — Siapkan daftar kalimat

Edit `a_list` di bagian `__main__`:

```python
a_list = [
    'Halo, apa kabar?',
    'Model transformer sangat powerful.',
    # ... ribuan kalimat
]
```

Atau muat dari file:

```python
with open('kalimat.txt', encoding='utf-8') as f:
    a_list = [l.strip() for l in f if l.strip()]
```

### Langkah 2 — Atur ukuran batch

```python
combined_length = 50   # kurangi jika total karakter > 5000
```

### Langkah 3 — Jalankan

```bash
cd data_augmentation
python 003-google_trans_data_aug.py
```

Pastikan `len(a_list) == len(a_trans_list)` (assert di akhir skrip).

### Langkah 4 — Back-translation dua arah

1. Pertama: `sl=zh-CN`, `tl=en` (atau `id` → `en` jika didukung).  
2. Kedua: terjemahkan hasil kembali ke bahasa asal.  
3. Simpan hanya jika hasil ≠ input.

---

## 5. Metrik evaluasi

| Metrik | Cara |
|--------|------|
| **Success rate API** | `success / (len(a_list)//combined_length)` |
| **Alignment batch** | `len(train_result) == len(content.split('\n'))` — harus selalu benar |
| **Kualitas terjemahan** | Sampling 30 pasang; skor 1–5 manual |
| **Dampak model** | F1/accuracy setelah menambah data augmentasi |

Log contoh di skrip: setiap 10 batch sukses mencetak 10 terjemahan terakhir — gunakan untuk audit cepat.

---

## 6. Deploy

### 6.1 Batch offline

Jalankan di mesin dengan akses internet; simpan `a_trans_list` ke CSV:

```python
import pandas as pd
pd.DataFrame({'asli': a_list, 'aug': a_trans_list}).to_csv('aug_google.csv', index=False)
```

### 6.2 Alternatif produksi

Untuk lingkungan stabil, pertimbangkan:

- [Google Cloud Translation API](https://cloud.google.com/translate) (resmi, berbayar)  
- Azure Translator, DeepL API  

Ganti `translate()` dengan klien resmi — struktur batch tetap sama.

### 6.3 Retry & rate limit

Skrip memakai `while get_trans: try/except` — tambahkan `time.sleep` lebih besar jika IP diblokir.

---

## 7. Troubleshooting

| Masalah | Solusi |
|---------|--------|
| `execjs` error | Pasang Node.js atau PyMiniRacer |
| `wrong` / exception loop | Kurangi `combined_length`; periksa karakter `\n` dalam kalimat |
| Respons kosong | Endpoint Google berubah; migrasi ke API resmi |
| `eval(result)` berbahaya | Ganti dengan `json.loads` jika respons valid JSON |
| Import `urllib.parse` | Pastikan `import urllib.parse` ada di skrip |

---

## Catatan etika & lisensi

Scraping web Google Translate dapat melanggar ketentuan layanan. Untuk penelitian kampus, dokumentasikan keterbatasan; untuk produksi gunakan **API resmi**.

---

## Referensi

- Sennrich et al., back-translation untuk NMT  
- [shawroad/NLP_pytorch_project](https://github.com/shawroad/NLP_pytorch_project)
