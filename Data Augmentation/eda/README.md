# Hands-On: Augmentasi Data dengan EDA (nlpcda)

**Mata kuliah:** Pemrosesan Bahasa Alami Berbasis Transformer  
**Program Studi Sains Data · Fakultas Sains · Institut Teknologi Sumatera**

**Skrip:** [`../001-run_eda.py`](../001-run_eda.py)  
**Library:** [nlpcda](https://github.com/649453932/nlpcda) — teknik **EDA** (*Easy Data Augmentation*) untuk NLP.

---

## 1. Teori dan konsep

### 1.1 Mengapa augmentasi data?

Model deep learning (termasuk Transformer) mudah **overfitting** jika data berlabel sedikit. Augmentasi menghasilkan variasi teks baru yang **mempertahankan label** (untuk klasifikasi/NER) atau meningkatkan keragaman dialog, sehingga model lebih **general**.

### 1.2 Teknik EDA dalam modul ini

| Teknik | Fungsi di skrip | Ide |
|--------|------------------|-----|
| Karakter/setara | `EquivalentChar` | Ganti karakter dengan varian setara (mis. angka/huruf mirip) |
| Entitas acak | `Randomword` | Ganti entitas (perusahaan, dll.) dari daftar eksternal |
| Sinonim | `Similarword` | Ganti kata dengan sinonim |
| Homofon | `Homophone` | Ganti dengan kata homofon / mirip bunyi |
| Hapus karakter | `RandomDeleteChar` | Hapus karakter secara acak |
| Tukar posisi | `CharPositionExchange` | Tukar urutan karakter/kata |
| Terjemahan bolak-balik | `baidu_translate` (opsional) | Zh→En→Zh via API Baidu |

### 1.3 Alur dalam pipeline NLP

```
Korpus asli  →  Augmentasi (EDA)  →  Korpus diperbesar  →  Latih model (BERT, dll.)
                      ↓
              Evaluasi: F1/accuracy naik? overfitting turun?
```

Augmentasi **bukan** pengganti data asli berkualitas; kombinasikan dengan data bersih dan validasi yang ketat.

---

## 2. Arsitektur / struktur kode

```
001-run_eda.py
├── test_EquivalentChar()      → EquivalentChar.replace()
├── test_Randomword()          → Randomword.replace()
├── test_Similarword()         → Similarword.replace()
├── test_Homophone()           → Homophone.replace()
├── test_RandomDeleteChar()  → RandomDeleteChar.replace()
├── test_CharPositionExchange()
└── test_baidu_translate()     → baidu_translate (2 arah)
```

Setiap fungsi menerima:

- `test_str` — teks sumber  
- `create_num` — jumlah variasi yang dihasilkan  
- `change_rate` — proporsi token/karakter yang diubah  

---

## 3. Persiapan lingkungan

```bash
pip install nlpcda
# Jika gagal dari PyPI utama:
# pip install nlpcda -i https://pypi.douban.com/simple/
```

Untuk **Randomword** / entitas, library dapat membutuhkan berkas di `extdata/` (mis. `company.txt`) — ikuti dokumentasi nlpcda.

---

## 4. Langkah hands-on

### Langkah 1 — Uji semua teknik (demo)

```bash
cd data_augmentation
python 001-run_eda.py
```

Perhatikan keluaran per bagian: penggantian karakter, sinonim, penghapusan, dll.

### Langkah 2 — Integrasi ke dataset latih

Contoh loop untuk file teks satu label per baris:

```python
from nlpcda import Similarword

aug = Similarword(create_num=2, change_rate=0.2)
with open('train.txt', encoding='utf-8') as f:
    for line in f:
        teks, label = line.strip().rsplit('\t', 1)
        for varian in aug.replace(teks):
            print(varian, label, sep='\t')
```

### Langkah 3 — Pilih teknik sesuai tugas

| Tugas NLP | Teknik yang sering cocok |
|-----------|---------------------------|
| Klasifikasi sentimen | Sinonim, hapus karakter (ringan) |
| NER | Hati-hati: jangan ubah entitas sembarangan; pakai `Ner` dari nlpcda jika tersedia |
| Chatbot | Back-translation + EDA ringan |
| Teks pendek | `change_rate` kecil (0.1–0.2) |

### Langkah 4 — Back-translation di EDA (opsional)

Edit `test_baidu_translate`: isi `appid` dan `secretKey` Baidu Translate, lalu uji. Alur: **Indonesia → Inggris → Indonesia** (ubah `t_from` / `t_to`).

---

## 5. Metrik evaluasi

Augmentasi dievaluasi **tidak** dengan loss augmentasi, melainkan dampaknya pada model downstream:

| Metrik | Cara ukur |
|--------|-----------|
| **Akurasi / F1 validasi** | Latih model dengan vs tanpa data augmentasi |
| **Gap train–valid** | Gap mengecil → overfitting berkurang |
| **Konsistensi label** | Sampling manual: apakah label masih benar setelah augmentasi? |
| **Diversity** | Distinct-n, rata-rata jarak edit (Levenshtein) antar sampel baru |

**Eksperimen minimal untuk laporan:**

1. Baseline: latih hanya data asli (N sampel).  
2. +EDA: tambah 2× variasi per sampel dengan `Similarword`.  
3. Bandingkan F1 pada set uji yang **tidak** di-augmentasi.

---

## 6. Deploy / operasional

### 6.1 Batch offline (disarankan)

Jalankan skrip augmentasi di server/CI **sebelum** pelatihan; simpan hasil ke `train_aug.txt`. Model produksi hanya membaca data final — tidak perlu nlpcda saat inferensi.

```bash
python augment_pipeline.py  # buat wrapper Anda sendiri berdasarkan 001-run_eda.py
```

### 6.2 On-the-fly (latihan)

Gunakan `Dataset` PyTorch yang memanggil `Similarword.replace()` di `__getitem__` — augmentasi setiap epoch (lebih lambat, lebih beragam).

### 6.3 Produksi

- **Jangan** augmentasi saat inferensi pengguna akhir.  
- Simpan seed random agar eksperimen dapat direproduksi.  
- Log versi nlpcda dan parameter `change_rate`.

---

## 7. Parameter dan tips

| Parameter | Rentang disarankan | Catatan |
|-----------|-------------------|---------|
| `create_num` | 1–5 | Terlalu besar → duplikasi berlebihan |
| `change_rate` | 0.1–0.3 | Tinggi → teks tidak natural |
| Bahasa | — | Contoh asli Mandarin; untuk Indonesia sesuaikan atau uji manual kualitas sinonim |

---

## 8. Troubleshooting

| Masalah | Solusi |
|---------|--------|
| `ModuleNotFoundError: nlpcda` | `pip install nlpcda` |
| Sinonim tidak masuk akal | Turunkan `change_rate`; ganti ke hapus karakter / back-translation |
| Entitas salah | Hindari `Randomword` untuk NER; gunakan augmentasi khusus entitas |
| API translate gagal | Pakai modul [backtranslate_baidu](../backtranslate_baidu/README.md) |

---

## Referensi

- Wei & Zou, *EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks* (2019)  
- Kode asli: [shawroad/NLP_pytorch_project](https://github.com/shawroad/NLP_pytorch_project)
