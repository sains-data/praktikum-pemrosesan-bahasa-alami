# Hands-On: Chatbot Seq2Seq + Luong Attention

**Mata kuliah:** Pemrosesan Bahasa Alami Berbasis Transformer  
**Program Studi Sains Data · Fakultas Sains · Institut Teknologi Sumatera**

Modul ini membangun chatbot klasik **encoder–decoder** dengan **GRU** dan mekanisme perhatian **Luong**. Ini adalah baseline sebelum Transformer: membantu memahami alur seq2seq, teacher forcing, dan attention.

---

## 1. Teori dan konsep

### 1.1 Sequence-to-sequence (seq2seq)

Model memetakan urutan input (pertanyaan) ke urutan output (jawaban):

```
Pertanyaan:  w1  w2  w3  ...  →  Encoder GRU  →  konteks + hidden
Jawaban:     <SOS> y1 y2 ... <EOS>  ←  Decoder GRU (autoregresif)
```

### 1.2 Arsitektur dalam proyek ini

```
                    ┌──────────────────┐
  input tokens ───► │ Encoder GRU      │  2 lapisan, bidirectional
                    │ (embedding)      │
                    └────────┬─────────┘
                             │ encoder_outputs (semua langkah waktu)
                             ▼
                    ┌──────────────────┐
  <SOS>, y_{t-1} ─► │ Decoder GRU      │  1 lapisan
                    │ + Luong Attention│  dot / general / concat
                    └────────┬─────────┘
                             ▼
                    Linear → softmax → token berikutnya
```

**Berkas utama:**

| Berkas | Peran |
|--------|--------|
| `seq2seq.py` | `EncoderRNN`, `LuongAttnDecoderRNN`, `GreedySearchDecoder` |
| `data_helper.py` | Kosakata, normalisasi, batching, padding, mask loss |
| `train.py` | Loop pelatihan + penyimpanan checkpoint |
| `inference.py` | Muat model + greedy decoding interaktif |
| `config.py` | Hyperparameter |

### 1.3 Luong attention

Pada setiap langkah dekoder, vektor query (hidden dekoder) dibandingkan dengan semua hidden encoder untuk membentuk **bobot perhatian**, lalu konteks = bobot × keluaran encoder. Variasi di `config.py`: `attn_model = 'dot'`.

### 1.4 Teacher forcing

Saat latih, dengan probabilitas 0,3 model memakai **token benar** langkah sebelumnya sebagai input dekoder (teacher forcing); selain itu memakai prediksi sendiri. Ini mempercepat konvergensi tetapi menciptakan *exposure bias*.

---

## 2. Data dan lingkungan

### 2.1 Format data

File `data/chatbot.txt`: satu pasangan per baris, dipisah **tab**:

```text
hello how are you	i am fine thanks
what is your name	my name is bot
```

Data contoh dalam repo berbahasa Inggris (film dialog). Untuk praktikum Indonesia, siapkan file dengan format yang sama.

### 2.2 Dependensi

```bash
pip install torch
```

### 2.3 Token khusus

| ID | Token | Fungsi |
|----|-------|--------|
| 0 | PAD | Padding |
| 1 | SOS | Awal jawaban |
| 2 | EOS | Akhir jawaban |

---

## 3. Langkah hands-on

### Langkah 1 — Eksplorasi data

```bash
cd Chatbot/seq2seq_luong
python data_helper.py
```

Perhatikan log: pembersihan, jumlah pasangan, ukuran kosakata.

### Langkah 2 — Pelatihan

```bash
python train.py
```

- Total iterasi: `Config.total_step` (default 100.000).
- Setiap 4.000 langkah: checkpoint di `save_model/{iter}_checkpoint_model.tar`.
- Loss: **NLL masked** (hanya token non-PAD).

### Langkah 3 — Inferensi

1. Sesuaikan `loadFilename` di `inference.py` ke checkpoint yang ada (mis. `4000_checkpoint_model.tar`).
2. Jalankan:

```bash
python inference.py
```

Secara default skrip memakai kalimat uji tetap; ubah loop `input()` di `evaluateInput` untuk chat interaktif penuh.

### Langkah 4 — Eksperimen (untuk laporan)

| Eksperimen | Variabel |
|------------|----------|
| Panjang kalimat | `MAX_LENGTH` (default 10 — sangat pendek, naikkan untuk dialog nyata) |
| Teacher forcing | `teacher_forcing_ratio` di `train.py` |
| Attention | `attn_model`: `dot`, `general`, `concat` |
| Ukuran hidden | `hidden_size` |

---

## 4. Metrik evaluasi

### 4.1 Saat pelatihan

| Metrik | Definisi |
|--------|----------|
| **Masked NLL loss** | Negative log-likelihood rata-rata pada posisi target yang bukan PAD |
| **Persentase iterasi** | Dicetak di log setiap batch |

Fungsi `maskNLLLoss` di `train.py`:

- Mengabaikan posisi padding lewat `mask`.
- Menggunakan `torch.gather` pada distribusi softmax dekoder.

### 4.2 Evaluasi kualitatif & kuantitatif

1. **Perplexity (perkiraan)** — `exp(loss)` pada set validasi (tambahkan loop validasi jika diperlukan).
2. **BLEU** — bandingkan hipotesis greedy dengan jawaban referensi.
3. **Response appropriateness** — penilaian manual 1–5 untuk 30 pertanyaan uji.

Contoh skor log probabilitas dari `GreedySearchDecoder` (tersedia di `inference.py` sebagai `scores`).

### 4.3 Checklist laporan

- [ ] Grafik loss vs iterasi
- [ ] Contoh 5 dialog: input, output model, referensi
- [ ] Pengaruh `MAX_LENGTH` dan `MIN_COUNT` pada kosakata
- [ ] Perbandingan singkat dengan modul `transformer_chatbot` (minggu berikutnya)

---

## 5. Deploy model

### 5.1 Muat checkpoint

```python
checkpoint = torch.load('save_model/4000_checkpoint_model.tar', map_location='cpu')
encoder.load_state_dict(checkpoint['encoder'])
decoder.load_state_dict(checkpoint['decoder'])
voc.__dict__ = checkpoint['voc_dict']
encoder.eval()
decoder.eval()
searcher = GreedySearchDecoder(encoder, decoder)
```

### 5.2 API Flask (contoh)

```python
from flask import Flask, request, jsonify
# ... muat encoder, decoder, voc, searcher ...

@app.route('/chat', methods=['POST'])
def chat():
    kalimat = normalizeString(request.json['text'])
    kata = evaluate(encoder, decoder, searcher, voc, kalimat)
    kata = [w for w in kata if w not in ('EOS', 'PAD')]
    return jsonify({'reply': ' '.join(kata)})
```

### 5.3 Catatan produksi

- Simpan `voc_dict` bersama model agar kosakata konsisten.
- Normalisasi input harus sama dengan pelatihan (`normalizeString`).
- Untuk bahasa Indonesia, ubah regex di `normalizeString` agar tidak hanya `a-zA-Z`.

---

## 6. Hyperparameter (`config.py`)

| Parameter | Default | Saran praktikum |
|-----------|---------|-----------------|
| `MAX_LENGTH` | 10 | Naikkan ke 20–40 untuk kalimat lebih panjang |
| `MIN_COUNT` | 3 | Frekuensi minimum kata di kosakata |
| `hidden_size` | 500 | — |
| `encoder_n_layers` | 2 | Encoder bidirectional |
| `decoder_n_layers` | 2 | Hidden awal diambil dari encoder |
| `batch_size` | 5 | — |
| `total_step` | 100000 | Kurangi untuk uji cepat (mis. 5000) |
| `clip` | 50.0 | Gradient clipping |

---

## 7. Troubleshooting

| Masalah | Solusi |
|---------|--------|
| `KeyError: unknown word` | Kata tidak ada di kosakata; turunkan `MIN_COUNT` atau perluas data |
| Loss tidak turun | Naikkan `total_step`, periksa format tab pada data |
| Jawaban hanya EOS/PAD | Model belum cukup latih; gunakan checkpoint lebih besar |
| CUDA error | Perbaiki `device` di `config.py` (`cuda:0` tanpa spasi) |

---

## Referensi

- Luong et al., *Effective Approaches to Attention-based Neural Machine Translation* (2015)  
- Sutskever et al., *Sequence to Sequence Learning with Neural Networks* (2014)  
- Kode asli: [shawroad/NLP_pytorch_project](https://github.com/shawroad/NLP_pytorch_project) — `Chatbot/seq2seq_luong`
