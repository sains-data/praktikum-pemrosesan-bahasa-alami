# Hands-On: Chatbot Berbasis BERT (Gaya UniLM)

**Mata kuliah:** Pemrosesan Bahasa Alami Berbasis Transformer  
**Program Studi Sains Data · Fakultas Sains · Institut Teknologi Sumatera**

Modul ini mengimplementasikan chatbot **sequence-to-sequence** dengan encoder **BERT** dan kepala prediksi bahasa (language modeling head). Pola pelatihannya mirip **UniLM**: satu urutan gabungan (pertanyaan + jawaban), dengan mask perhatian khusus agar model hanya memprediksi bagian jawaban.

---

## 1. Teori dan konsep

### 1.1 Masalah yang diselesaikan

Diberikan pasangan dialog `(pertanyaan, jawaban)`, model belajar memprediksi token jawaban berdasarkan pertanyaan. Ini adalah tugas **conditional language modeling** pada format dua segmen:

```
[CLS] pertanyaan [SEP] jawaban [SEP]
```

- Segmen A (`token_type_ids = 0`): pertanyaan pengguna  
- Segmen B (`token_type_ids = 1`): jawaban yang harus diprediksi  

### 1.2 Arsitektur

```
┌─────────────────────────────────────────────────────────┐
│  Input: token_ids + token_type_ids + attention mask     │
└──────────────────────────┬──────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────┐
│  BERT Encoder (multi-layer Transformer)                 │
│  · Self-attention + FFN per lapisan                     │
│  · Mask: hanya melihat token yang diizinkan (causal     │
│    pada segmen jawaban, lihat seq2seq_bert.py)         │
└──────────────────────────┬──────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────┐
│  BertLMPredictionHead → logits per posisi (vocab_size)  │
└──────────────────────────┬──────────────────────────────┘
                           ▼
              Cross-entropy (hanya posisi segmen B)
```

**Komponen utama (berkas):**

| Berkas | Peran |
|--------|--------|
| `bert_model.py` | Implementasi BERT (embedding, attention, layer norm) |
| `seq2seq_bert.py` | Gabungan encoder + decoder head, loss, `generate()`, beam search |
| `tokenizer.py` | Tokenisasi gaya BERT (WordPiece) |
| `dataloader.py` | Membaca pasangan `pertanyaan=jawaban`, padding dinamis |
| `train.py` | Loop pelatihan + penyimpanan checkpoint |
| `interface.py` | Inferensi interaktif (chat di terminal) |

### 1.3 Pra-pelatihan dan fine-tuning

- Bobot **BERT pra-latih** (Bahasa Mandarin) dimuat dari `data/pytorch_model.bin`.
- Lapisan atas dilatih ulang pada korpus dialog Anda (`data/chatbot.txt`).
- Ini mengurangi kebutuhan data dibanding melatih Transformer dari nol.

### 1.4 Decoding saat inferensi

- **`generate()`**: memperluas urutan token demi token pada segmen B.
- **`beam_search`**: menjaga beberapa hipotesis jawaban; `beam_size` mengontrol lebar pencarian.
- Berhenti saat token `[SEP]` atau mencapai `out_max_length`.

---

## 2. Persiapan data dan lingkungan

### 2.1 Format data

File `data/chatbot.txt`, satu dialog per baris:

```text
Apa kabar?=Baik, terima kasih.
Cuaca hari ini?=Cerah dan hangat.
```

Pemisah pasangan: tanda `=` (bukan tab).

### 2.2 Berkas yang harus ada

| Berkas | Keterangan |
|--------|------------|
| `data/chatbot.txt` | Korpus dialog (siapkan sendiri atau konversi dari dataset kursus) |
| `data/vocab.txt` | Kosakata BERT (sudah disertakan) |
| `data/pytorch_model.bin` | Checkpoint BERT pra-latih (unduh dari repositori BERT-Chinese resmi) |

### 2.3 Dependensi

```bash
pip install torch pandas
```

Sesuaikan `Config.device` di `config.py` (`cuda` atau `cpu`).

---

## 3. Langkah hands-on

### Langkah 1 — Pahami alur data

1. Buka `dataloader.py`: fungsi `read_corpus` memecah baris pada `=`.
2. `Tokenizer.encode(src, tgt)` menghasilkan `token_ids` dan `token_type_ids`.
3. `collate_fn` melakukan padding per batch.

### Langkah 2 — Pelatihan

```bash
cd Chatbot/Bert_chatbot
python train.py
```

Yang terjadi:

1. Memuat dataset `DreamDataset`.
2. Membangun `Seq2SeqModel` dan memuat bobot BERT pra-latih.
3. Optimizer Adam, learning rate `1e-5` (lihat `config.py`).
4. Setiap 30 langkah: menyimpan `bert_dream.bin`.
5. Setiap 10 epoch: menyimpan `data/pytorch_bert_gen_epoch{N}.bin`.

### Langkah 3 — Pantau pelatihan

- Metrik utama saat latih: **loss cross-entropy** (rata-rata per epoch dicetak di akhir epoch).
- Loss hanya dihitung pada token jawaban (mask dari `token_type_ids`).

### Langkah 4 — Inferensi interaktif

```bash
python interface.py
```

Pastikan `bert_dream.bin` ada di folder kerja. Ketik pertanyaan; model menampilkan jawaban.

Opsional di kode:

```python
reply = bert_seq2seq.generate(text, beam_size=3)
```

---

## 4. Metrik evaluasi

### 4.1 Saat pelatihan

| Metrik | Arti | Target praktikum |
|--------|------|------------------|
| **Training loss** | CE pada token jawaban (tanpa PAD/segmen A) | Turun stabil antar-epoch |
| **Loss per step** | Dicetak setiap batch (`train.py`) | Tidak NaN; tidak meledak |

### 4.2 Evaluasi kualitatif (disarankan untuk laporan)

Karena modul asli tidak menyertakan skrip evaluasi terpisah, gunakan:

1. **Sampel dialog manual** — 20–50 pasangan uji yang tidak ikut latih.
2. **BLEU / ROUGE** (opsional) — bandingkan jawaban model dengan referensi jika format pasangan tetap sama.
3. **Kriteria subjektif**: relevansi, kelancaran, tidak mengulang token.

Contoh perhitungan BLEU sederhana (opsional):

```python
# pip install nltk
from nltk.translate.bleu_score import sentence_bleu
ref = [["jawaban", "referensi"]]
hyp = ["jawaban", "model"]
print(sentence_bleu(ref, hyp))
```

### 4.3 Checklist laporan

- [ ] Kurva loss (screenshot log atau TensorBoard jika ditambahkan)
- [ ] 5 contoh dialog: input → jawaban model → komentar
- [ ] Pengaruh `beam_size` (1 vs 3)
- [ ] Batasan: ketergantungan pada BERT Mandarin jika data Indonesia

---

## 5. Deploy model

### 5.1 Deploy lokal (terminal)

Sudah tersedia via `interface.py`:

```bash
python interface.py
```

Muat checkpoint:

```python
checkpoint = torch.load('./bert_dream.bin', map_location='cpu')
bert_seq2seq.load_state_dict(checkpoint)
bert_seq2seq.eval()
```

### 5.2 Deploy sebagai API sederhana (Flask)

Buat berkas `serve.py` (contoh untuk praktikum):

```python
from flask import Flask, request, jsonify
import torch
from seq2seq_bert import Seq2SeqModel
from bert_model import BertConfig
from tokenizer import load_bert_vocab

app = Flask(__name__)
word2idx = load_bert_vocab()
model = Seq2SeqModel(BertConfig(len(word2idx)))
model.load_state_dict(torch.load('bert_dream.bin', map_location='cpu'))
model.eval()

@app.route('/chat', methods=['POST'])
def chat():
    teks = request.json.get('text', '')
    jawaban = model.generate(teks, beam_size=3, device='cpu')
    return jsonify({'reply': jawaban})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

```bash
pip install flask
python serve.py
# curl -X POST http://localhost:5000/chat -H "Content-Type: application/json" -d '{"text":"halo"}'
```

### 5.3 Deploy produksi (ringkas)

| Tahap | Saran |
|-------|--------|
| Ekspor | Simpan `state_dict` atau TorchScript setelah `eval()` |
| Serving | TorchServe, FastAPI + worker GPU, atau ONNX Runtime |
| Skala | Batch inference, batasi `max_length`, cache tokenizer |

---

## 6. Konfigurasi penting (`config.py`)

| Parameter | Default | Keterangan |
|-----------|---------|------------|
| `batch_size` | 2 | Naikkan jika GPU memadai |
| `learning_rate` | 1e-5 | Tipikal fine-tuning BERT |
| `max_length` | 100 | Panjang maksimum urutan |
| `EPOCH` | 1000 | Kurangi untuk uji cepat (mis. 3–5) |

---

## 7. Troubleshooting

| Masalah | Solusi |
|---------|--------|
| File `pytorch_model.bin` tidak ada | Unduh BERT-Chinese; letakkan di `data/` |
| OOM GPU | Kurangi `batch_size` atau `max_length` |
| Jawaban kosong / `[UNK]` dominan | Periksa format `chatbot.txt` dan kosakata |
| `interface.py` error load | Pastikan path `bert_dream.bin` benar |

---

## Referensi

- Devlin et al., *BERT* (2018)  
- Dong et al., *Unified Language Model Pre-training* (UniLM)  
- Kode asli: [shawroad/NLP_pytorch_project](https://github.com/shawroad/NLP_pytorch_project) — folder `Chatbot/Bert_chatbot`
