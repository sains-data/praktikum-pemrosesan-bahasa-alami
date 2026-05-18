# Hands-On: Augmentasi Back-Translation (Baidu Translate)

**Mata kuliah:** Pemrosesan Bahasa Alami Berbasis Transformer  
**Program Studi Sains Data · Fakultas Sains · Institut Teknologi Sumatera**

**Skrip:** [`../002-run_contrslate_data_aug.py`](../002-run_contrslate_data_aug.py)

---

## 1. Teori dan konsep

### 1.1 Back-translation

Ide: terjemahkan teks ke bahasa pivot (mis. **Inggris**), lalu terjemahkan kembali ke bahasa asal. Kalimat hasil mempertahankan makna kasar tetapi **parafrase** struktur dan leksikon → data latih lebih beragam.

```
Teks asli (ID)  →  EN  →  Teks augmentasi (ID, parafrase)
```

Dipopulerkan untuk NMT dan klasifikasi teks ketika data berlabel terbatas.

### 1.2 Peran API Baidu

Skrip memanggil **Baidu Translate API** (berbayar/kuota gratis) dengan tanda tangan MD5 (`appid + teks + salt + secretKey`). Dua panggilan berurutan = satu siklus back-translation.

### 1.3 Format data contoh

File `q_gov.txt` (siapkan sendiri):

```text
pertanyaan asli||jawaban
berapa lama pengiriman?||3-5 hari kerja
```

Keluaran `q_gov_aug.txt`:

```text
pertanyaan asli||pertanyaan setelah back-trans||jawaban
```

---

## 2. Arsitektur kode

```
002-run_contrslate_data_aug.py
├── baidu_translate(content, appid, secretKey, t_from, t_to)
│     ├── Validasi panjang (< 4891 karakter)
│     ├── Hitung sign MD5
│     └── GET api.fanyi.baidu.com/.../translate
└── __main__: baca q_gov.txt → augment 100 baris pertama → tulis q_gov_aug.txt
```

---

## 3. Persiapan

### 3.1 Akun API

1. Daftar di [Baidu Translate Open Platform](http://api.fanyi.baidu.com/api/trans/product/desktop).  
2. Dapatkan **APP ID** dan **Secret Key**.  
3. Isi di skrip (jangan commit ke repo publik).

### 3.2 Dependensi

```bash
pip install requests tqdm
```

### 3.3 File input

Letakkan `q_gov.txt` di folder `data_augmentation/` (satu baris per pasangan `teks||label`).

---

## 4. Langkah hands-on

### Langkah 1 — Konfigurasi kredensial

Edit bagian `__main__` di `002-run_contrslate_data_aug.py`:

```python
temp = baidu_translate(content=ori_question, appid='APP_ID_ANDA',
                       secretKey='SECRET_ANDA', t_from='id', t_to='en')
trans_question = baidu_translate(content=temp, appid='APP_ID_ANDA',
                               secretKey='SECRET_ANDA', t_from='en', t_to='id')
```

> Kode asli memakai `zh`/`en`; untuk Indonesia gunakan kode bahasa yang didukung API (mis. `id`, `en` — verifikasi di dokumentasi Baidu).

### Langkah 2 — Jalankan

```bash
cd data_augmentation
python 002-run_contrslate_data_aug.py
```

### Langkah 3 — Perluas ke seluruh file

Ubah `lines[:100]` menjadi `lines` setelah uji 10–100 baris sukses.

### Langkah 4 — Gabung ke pipeline latih

- Gunakan `trans_question` sebagai sampel tambahan dengan **label sama** dengan `ans`.  
- Buang pasangan jika terjemahan identik dengan asli (tidak menambah informasi).

---

## 5. Metrik evaluasi

| Metrik | Keterangan |
|--------|------------|
| **Tingkat parafrase** | % sampel augmentasi ≠ teks asli (harus > 0) |
| **BLEU asli–augmentasi** | Terlalu tinggi → augmentasi lemah; terlalu rendah → makna rusak |
| **F1 / accuracy model** | Bandingkan latih dengan/tanpa baris augmentasi |
| **Biaya & latensi** | Jumlah karakter × 2 panggilan API per sampel |

**Quality check manual:** periksa 50 sampel — apakah label masih benar?

---

## 6. Deploy

### 6.1 Batch offline (disarankan)

```bash
# Cron / workflow sekali sehari
python 002-run_contrslate_data_aug.py
```

Simpan `q_gov_aug.txt` ke storage; pipeline ML membaca versi ter-augmentasi.

### 6.2 Keamanan

- Simpan `APP_ID` / `SECRET` di variabel lingkungan:

```python
import os
appid = os.environ['BAIDU_APP_ID']
secret = os.environ['BAIDU_SECRET_KEY']
```

### 6.3 Produksi

- Batasi rate request (sleep antar baris, batch).  
- Retry dengan backoff jika API error.  
- **Tidak** panggil API translate saat inferensi model akhir.

---

## 7. Troubleshooting

| Masalah | Solusi |
|---------|--------|
| Error sign / 54001 | Periksa APP ID, secret, encoding UTF-8 |
| `输入请不要超过4891个字符` | Potong teks panjang per chunk |
| Terjemahan sama persis | Coba bahasa pivot lain; naikkan variasi dengan EDA |
| File tidak ditemukan | Buat `q_gov.txt` atau ubah path |

---

## Referensi

- Sennrich et al., *Improving Neural Machine Translation Models with Monolingual Data* (back-translation)  
- [shawroad/NLP_pytorch_project](https://github.com/shawroad/NLP_pytorch_project)
