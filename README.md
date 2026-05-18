# Panduan Hands-On: Pemrosesan Bahasa Alami Berbasis Transformer

**Mata kuliah:** Pemrosesan Bahasa Alami Berbasis Transformer  
**Program studi:** Sains Data  
**Fakultas:** Sains  
**Institusi:** Institut Teknologi Sumatera (Itera)

Repositori ini digunakan sebagai **laboratorium praktikum** untuk mempelajari berbagai tugas NLP dengan implementasi **PyTorch**, dengan penekanan pada arsitektur **Transformer** dan turunannya (BERT, GPT, seq2seq Transformer, dan sejenisnya).

---

## Kredit dan sumber

Kode dan struktur proyek diambil dari repositori open source berikut:

> **[shawroad/NLP_pytorch_project](https://github.com/shawroad/NLP_pytorch_project)**  
> Koleksi implementasi PyTorch untuk embedding, klasifikasi teks, NER, NMT, generasi teks, chatbot, distilasi model, dan tugas NLP lainnya.

README asli (bahasa Mandarin) disimpan di [`README-old.md`](README-old.md) sebagai referensi. Panduan ini adalah **adaptasi hands-on dalam Bahasa Indonesia** khusus untuk keperluan mata kuliah di Itera, tanpa mengubah hak cipta kode sumber.

Penulis repositori asli juga memecah beberapa tugas ke repositori terpisah (misalnya [Text-Classification-Pytorch](https://github.com/shawroad/Text-Classification-Pytorch), [Semantic-Textual-Similarity-Pytorch](https://github.com/shawroad/Semantic-Textual-Similarity-Pytorch), [Text-Generation-Chinese-Pytorch](https://github.com/shawroad/Text-Generation-Chinese-Pytorch)). Anda boleh merujuk ke sana jika ingin eksperimen perbandingan model yang lebih terfokus.

---

## Tujuan pembelajaran

Setelah menyelesaikan praktikum di repositori ini, mahasiswa diharapkan mampu:

1. Menjelaskan alur kerja umum NLP: data → representasi → model → latih → evaluasi → inferensi.
2. Mengimplementasikan dan menjalankan model berbasis **self-attention / Transformer** pada tugas klasifikasi, pelabelan urutan, terjemahan, dan generasi.
3. Membandingkan pendekatan **pra-pelatihan + fine-tuning** (mis. BERT) dengan model yang dilatih dari awal atau arsitektur klasik (CNN, BiLSTM).
4. Membaca struktur kode per modul dan menyesuaikan hyperparameter, data, serta skrip latih.

---

## Persiapan lingkungan
### Perangkat lunak

| Komponen | Rekomendasi |
|----------|-------------|
| Python | 3.8 – 3.10 |
| PyTorch | Sesuai CUDA/CPU mesin Anda |
| GPU | Disarankan untuk modul BERT, GPT, NMT Transformer |

### Instalasi dasar

```bash
# Buat lingkungan virtual (opsional tetapi disarankan)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Dependensi umum (sesuaikan versi jika diperlukan)
pip install torch torchvision torchaudio
pip install transformers numpy pandas scikit-learn tqdm
```

> **Catatan:** Setiap subfolder dapat membutuhkan paket tambahan (mis. `crf`, `nlpcda`, `jieba`). Baca komentar di awal skrip `train.py` atau file `readme.txt` / `readme.md` di folder modul yang dipilih.

### Struktur repositori

Setiap modul utama memiliki **`README.md`** panduan hands-on (Bahasa Indonesia). Nama folder di bawah ini adalah path aktual di repositori.

```
NLP-Projects/
├── Chatbot/                      # Chatbot: BERT, seq2seq, Transformer
├── Demo-day/                     # Demo dev + deploy chatbot → FastAPI → Discord
├── Data-Augmentation/            # EDA, back-translation
├── Distillation/                 # Distilasi & kompresi model BERT
├── Embedding/                    # Word2Vec, BERT, ALBERT, NPLM
├── Image-Caption/                # Caption gambar (ResNet+RNN, CLIP)
├── Keyword/                      # TF-IDF, TextRank, KeyBERT
├── Large-Language-Model/         # LoRA ChatGLM, PEFT, Accelerate
├── Latent-Dirichlet-Allocation/  # Pemodelan topik LDA
├── Named-Entity-Recognition/     # NER: BiLSTM-CRF, BERT-CRF, GlobalPointer
├── Neural-Machine-Translation/   # Seq2seq GRU/LSTM, Transformer NMT
├── Optical-Character-Recognition/ # OCR subtitle video
├── Pretrain-Model/               # Pra-pelatihan BERT / WoBERT / MLM-only
├── Reading-Comprehension/        # Machine Reading Comprehension
├── Relation-Extraction/          # Ekstraksi relasi
├── Slot_Filling/                 # Intent + slot (JointBERT)
├── Text_Classification/          # Klasifikasi teks (BERT, RoBERTa, XLNet, …)
├── Text_Clustering/              # LDA, K-means, DBSCAN
├── Text_Corrector/               # Koreksi ejaan BERT
├── Text_Generation/              # GPT-2, T5, generasi teks
├── Text_Ranking/                 # BM25, DPR, re-ranking
├── Text_Similarity/              # SimCSE, ESIM, Bert Whitening, …
└── README-old.md                 # README asli (Mandarin)
```

---

## Cara kerja praktikum (hands-on)

Ikuti langkah berikut **untuk setiap modul** yang dikerjakan:

1. **Pilih modul** sesuai topik minggu / tugas (lihat [Jalur pembelajaran](#jalur-pembelajaran-yang-disarankan)).
2. **Masuk ke subfolder** implementasi (contoh: `Text_Classification/roberta_classification/`).
3. **Baca** file konfigurasi, `readme.txt`, dan skrip `train.py` / `model.py`.
4. **Siapkan data** — banyak modul menyertakan contoh kecil di `data/`; untuk tugas kursus, ganti dengan dataset Anda dan sesuaikan path di config.
5. **Jalankan pelatihan** dari direktori modul tersebut:
   ```bash
   cd Text_Classification/roberta_classification
   python train.py
   ```
6. **Jalankan inferensi** jika tersedia (`inference.py`, `chat.py`, dll.).
7. **Catat** di laporan: arsitektur, loss, metrik, dan perbandingan dengan baseline (mis. TextCNN vs BERT).

### Format laporan singkat (disarankan)

- Judul modul dan tugas NLP yang diselesaikan  
- Perintah yang dijalankan  
- Cuplikan loss / metrik evaluasi  
- Analisis singkat (2–3 paragraf): apa yang dilakukan Transformer, dan apa bedanya dengan modul non-Transformer di repo yang sama  

---

## Jalur pembelajaran yang disarankan

Urutan di bawah ini selaras dengan progres mata kuliah **berbasis Transformer**. Modul non-Transformer disertakan sebagai **baseline** atau konteks historis.

| Minggu (contoh) | Topik | Folder / modul | Fokus |
|-----------------|-------|----------------|-------|
| 1 | Representasi teks | `Embedding/` | Word2Vec → BERT/ALBERT |
| 2 | Klasifikasi teks | `Text_Classification/` | TextCNN, BiLSTM (baseline) → `roberta_classification`, `How_to_finetune_bert_classification` |
| 3 | Pelabelan urutan (NER) | `Named-Entity-Recognition/` | `BiLSTM_CRF_Ner` → `Bert_CRF_Ner`, `Bert_Softmax_Ner` |
| 4 | Arsitektur seq2seq & perhatian | `Neural-Machine-Translation/`, `Chatbot/seq2seq_luong` | GRU + attention → `Transformers_NMT`, `Chatbot/transformer_chatbot` |
| 5 | Pra-pelatihan & fine-tuning | `Pretrain-Model/` | `bert-pretrain`, konsep MLM |
| 6 | Kemiripan semantik | `Text_Similarity/` | `SimCSE_Unsupervised`, `Bert_Whitening`, `ESIM` |
| 7 | Generasi teks | `Text_Generation/` | `Simple-GPT2`, `GPT2_SummaryGen` |
| 8 | Pemahaman bacaan & dialog | `Reading-Comprehension/mrc_baseline`, `Chatbot/Bert_chatbot` | BERT untuk MRC & chatbot |
| 9 | Distilasi & efisiensi | `Distillation/tiny_bert` | Menyusutkan model guru (BERT) |
| 10 | Topik lanjutan (opsional) | `Large-Language-Model/`, `Text_Ranking/`, `Slot_Filling/` | LoRA, ranking, intent+slot |

Anda bebas menyesuaikan urutan dengan silabus dosen pengampu.

---

## Daftar modul dan perintah

Di bawah ini, setiap entri menjelaskan **tujuan modul** dan **perintah utama**. Jalankan perintah dari **direktori modul yang bersangkutan**, kecuali dinyatakan lain.

### Chatbot

| Modul | Deskripsi | Perintah |
|-------|-----------|----------|
| `Bert_chatbot` | Chatbot gaya UniLM dengan BERT | `python train.py` → `python infernece.py` |
| `seq2seq_luong` | Encoder 2×GRU, decoder 1×GRU + Luong attention | `python train.py` → `python inference.py` |
| `transformer_chatbot` | Chatbot **Transformer** standar; data contoh: Qingyun dialogue | `python train.py` → `python chat.py` |

### Distillation (distilasi model)

| Modul | Deskripsi | Perintah |
|-------|-----------|----------|
| `DynaBert` | Pruning struktur BERT (Huawei) | `python train_teacher_model.py` → `python train_tailor_model.py` |
| `rnn_distill_bert` | LSTM meniru keluaran BERT (soft label) | `python train_bert.py` → `python train_distill.py` |
| `three_layer_self-attention_to_distill_bert` | 3 lapisan Transformer encoder → distilasi BERT | `python train_bert.py` → `python train_distill.py` |
| `tiny_bert` | TinyBERT: soft label + MSE antar lapisan | `python train.py` → `python train_distill_v2.py` |

### Embedding (representasi kata)

| Berkas / modul | Deskripsi | Perintah |
|----------------|-----------|----------|
| `001-skipgram-word2vec.py` | Word2Vec (Skip-gram) | `python 001-skipgram-word2vec.py` |
| `002-bert.py` | Latih BERT dari awal / lanjutan | `python 002-bert.py` |
| `003-albert.py` | Latih ALBERT | `python 003-albert.py` |
| `004-NPLM.py` | Neural probabilistic LM (klasik) | `python 004-NPLM.py` |

### Named-Entity-Recognition (NER)

Panduan indeks: [`Named-Entity-Recognition/README.md`](Named-Entity-Recognition/README.md)

| Modul | Deskripsi | Perintah |
|-------|-----------|----------|
| `Bert_CRF_Ner` | BERT + CRF untuk sequence labeling | `python run_ner_crf.py` → `python inference.py` |
| `Bert_Softmax_Ner` | BERT + softmax per token | `python train.py` → `python inference.py` |
| `BiLSTM_CRF_Ner` | Baseline BiLSTM + CRF | `python train.py` |
| `GlobalPointer` | Pointer network untuk NER (lanjutan) | Lihat `GlobalPointer/data/README.md` |

Jalankan dari direktori modul, contoh:

```bash
cd Named-Entity-Recognition/Bert_CRF_Ner
python run_ner_crf.py
```

### Neural-Machine-Translation (NMT)

Panduan: [`Neural-Machine-Translation/README.md`](Neural-Machine-Translation/README.md)

| Modul / berkas | Deskripsi | Perintah |
|----------------|-----------|----------|
| `001-gru_seq2seq_attention.py` | Seq2seq GRU + attention (monolit) | `python 001-gru_seq2seq_attention.py` |
| `002-lstm_seq2seq_attention.py` | Seq2seq LSTM + attention | `python 002-lstm_seq2seq_attention.py` |
| `GRU_attention/` | GRU + attention (modular) | `python train.py` |
| `Transformers_NMT/` | **Transformer** untuk terjemahan mesin | `python process_data.py` → `python train.py` |

```bash
cd Neural-Machine-Translation/Transformers_NMT
python train.py
```

### Pretrain-Model (pra-pelatihan)

Panduan: [`Pretrain-Model/README.md`](Pretrain-Model/README.md)

| Modul | Deskripsi | Perintah |
|-------|-----------|----------|
| `bert-pretrain/` | MLM + NSP BERT | `python get_train_data.py` → `python run_pretrain.py` |
| `mask_pretrain/` | MLM saja (tanpa NSP) | `python run_pretrain.py` |
| `wobert-pretrain/` | WoBERT + whole-word mask | `python process_pretrain_data.py` → `python run_pretrain.py` |

### Reading-Comprehension (MRC)

| Modul | Deskripsi | Perintah |
|-------|-----------|----------|
| `BERT_MRC` | BERT untuk MRC (teks panjang: pemotongan) | `python train.py` |
| `BiDAF`, `DocQA`, `Match_LSTM`, `RNet` | Model klasik MRC | `python data_process.py` → skrip `train_*.py` |
| `QANet` | MRC tanpa RNN; self-attention awal di MRC | `python data_process.py` → `python train.py` |
| `mrc_baseline` | **Disarankan pertama** untuk MRC: sliding window, ranking jawaban, adversarial training | `python train.py` |
| `albert_mrc`, `roberta_mrc`, `electra_bert` | Fine-tuning ALBERT / RoBERTa / ELECTRA | Lihat skrip `train.py` / `run_cail.py` di masing-masing folder |
| `transformer+rnn+attention` | MRC generatif: encoder Transformer + decoder GRU | `python train.py` → `python inference.py` |
| `transformer_reading` | MRC generatif: **Transformer** penuh | `python train.py` → `python inference.py` |

### Slot_Filling

| Modul | Deskripsi | Perintah |
|-------|-----------|----------|
| `JointBert` | Klasifikasi intent (vektor CLS) + slot filling per token | `python train.py` |

### Text_Classification

| Modul | Deskripsi | Perintah |
|-------|-----------|----------|
| `DPCNN` | CNN dalam + residual | `python get_data_to_examples.py` → `python examples_to_features.py` → `python train.py` |
| `FastBert` | Self-distillation untuk inferensi cepat | `sh train_stage0.sh` → `sh train_stage1.sh` |
| `Fasttext` | FastText (Facebook) | `python step1_get_data_to_examples.py` → `step2` → `train.py` |
| `XLNet` | XLNet + ide Transformer-XL | `python train.py` |
| `all_layer_out_concat` | Gabungan CLS dari semua lapisan BERT + attention | `python train.py` → `python inference.py` |
| `bert+bceloss+average_checkpoint` | BCELoss + rata-rata bobot checkpoint | `python run_classify.py` → `python run_average_checkpoints.py` |
| `capsule_text_classification` | GRU + Capsule Network | `python train.py` |
| `longformer_classification` | Longformer untuk teks panjang | `python train.py` |
| `multi_label_classify_bert` | Klasifikasi multi-label (3 varian BERT) | `python train.py` → `python inference.py` |
| `roberta_classification` | Fine-tuning RoBERTa | `python train.py` |
| `transformer_xl` | Transformer-XL untuk teks panjang | `python train.py` |
| `wobert+focal_loss` | WoBERT + focal loss (kelas tidak seimbang) | `python run_classify.py` |
| `001-TextCNN.py`, `002-BILSTM+Attention.py` | Baseline di folder `Text_Classification/` | Jalankan berkas sesuai nama |

### Text_Clustering

| Modul | Perintah |
|-------|----------|
| LDA | `python train_LDA_cluster.py` |
| DBSCAN | `python train_dbscan_cluster.py` |
| K-means | `python train_kmeans_cluster.py` |

### Text_Corrector

| Modul | Deskripsi | Perintah |
|-------|-----------|----------|
| `bert_for_correction` | Koreksi ejaan token-level dengan BERT | `python run_pretrain_bert.py` → `python bert_corrector.py` |

### Text_Generation

| Modul | Deskripsi | Perintah |
|-------|-----------|----------|
| `GPT2_SummaryGen` | Ringkasan teks dengan GPT-2 | `python train.py` → `python inferface.py` |
| `GPT2_TitleGen` | Generasi judul artikel | `python train.py` → `python inference.py` |
| `Simple-GPT2` | Implementasi GPT-2 dari nol | `python train.py` → `python inference.py` |
| `T5_generation` | Generasi dengan T5 | Lihat skrip di folder modul |

### Text_Ranking

| Modul | Deskripsi | Perintah |
|-------|-----------|----------|
| `BM25` | Pemeringkatan leksikal BM25 | `python main.py` |
| `DC_Bert_Ranking` | Dual encoder + interaksi Transformer | `python train.py` → `python inference.py` |
| `DPR_Ranking` | Dense Passage Retrieval (Facebook) | `python train.py` |
| `MT_Ranking` | Encoding BERT-style + skor CLS | `python train.py` → `python inference.py` |
| `ReRank` | Re-ranking + distilasi | `python train.py` → `python train_distill.py` |

### Text_Similarity

| Modul | Deskripsi | Perintah |
|-------|-----------|----------|
| `ABCNN`, `BiMPM`, `DecomposableAttention`, `ESIM`, `RE2`, `SiaGRU` | Model kemiripan kalimat (beragam arsitektur) | `python train.py` |
| `SimCSE_*` | SimCSE: contrastive learning + dropout | `python train.py` di folder terkait |
| `BM25.py`, `TF_IDF.py` | Baseline statistik | `python BM25.py` / `python TF_IDF.py` |
| `Bert_Whitening` | Whitening embedding BERT tanpa latih ulang | `python run_bert_whitening.py` |

### Data-Augmentation

Panduan: [`Data-Augmentation/README.md`](Data-Augmentation/README.md)

| Modul | Deskripsi | Perintah |
|-------|-----------|----------|
| EDA | Augmentasi: sinonim, hapus karakter, swap, dll. | `python 001-run_eda.py` |
| Back-translation (Baidu) | Augmentasi via terjemahan bolak-balik | `python 002-run_contrslate_data_aug.py` |
| Back-translation (Google) | Sama, via Google Translate | `python 003-google_trans_data_aug.py` |

```bash
cd Data-Augmentation
python 001-run_eda.py
```

### Relation-Extraction

| Modul | Deskripsi | Perintah |
|-------|-----------|----------|
| `lstm_cnn_information_extract` | Ekstraksi informasi LSTM+CNN | `python train.py` → `python inference.py` |
| `relation_classification` | Klasifikasi relasi BiLSTM+attention | `python data_helper.py` → `python train.py` |

### Modul tambahan (panduan hands-on tersedia)

| Folder | Isi singkat | Panduan |
|--------|-------------|---------|
| `Large-Language-Model/` | LoRA ChatGLM, PEFT BERT, Accelerate | [README](Large-Language-Model/README.md) |
| `Image-Caption/` | ResNet+RNN, CLIP caption | [README](Image-Caption/README.md) |
| `Keyword/` | TF-IDF, TextRank, KeyBERT | [README](Keyword/README.md) |
| `Latent-Dirichlet-Allocation/` | Pemodelan topik LDA | [README](Latent-Dirichlet-Allocation/README.md) |
| `Optical-Character-Recognition/` | OCR subtitle video (`ekstraksi_subtitle_video/`) | [README](Optical-Character-Recognition/README.md) |
| `Chatbot/` | BERT, seq2seq, Transformer chatbot | `Bert_chatbot/`, `seq2seq_luong/`, `transformer_chatbot/` (masing-masing ada README) |
| `Demo-day/` | Demo 2 hari: hari 1 latih chatbot (UDC), hari 2 FastAPI + Discord | [README](Demo-day/README.md) |
| `Distillation/` | TinyBERT, DynaBERT, distilasi RNN→BERT | [README](Distillation/README.md) |
| `Embedding/` | Skip-gram, BERT, ALBERT, NPLM | [README](Embedding/README.md) |
| `Text_Classification/RoFormer_CLS`, `CAN`, `RDrop`, dll. | Varian klasifikasi tambahan | Lihat `Text_Classification/` |
| `Text_Similarity/ConSBERT`, `SentenceBert`, `ESimCSE`, … | Varian kemiripan semantik | Lihat `Text_Similarity/` |

---

## Tips praktikum

1. **Mulai dari modul kecil** — misalnya `Embedding/001-skipgram-word2vec.py` atau `Text_Classification/001-TextCNN.py` sebelum BERT penuh.
2. **Perhatikan path data** — banyak error berasal dari path relatif; pastikan working directory benar (contoh: `cd Neural-Machine-Translation/Transformers_NMT`).
3. **GPU memori** — kurangi `batch_size` di file config jika terjadi OOM.
4. **Bahasa data** — sebagian besar contoh menggunakan **teks Mandarin**; untuk tugas Indonesia, siapkan korpus sendiri dan sesuaikan tokenizer (mis. multilingual BERT atau model Indonesia).
5. **Versi library** — repositori asli dari 2023; jika `transformers` versi terbaru menimbulkan error API, catat versi yang berhasil di laporan.

---

## Kontribusi dan etika akademik

- Hormati lisensi dan atribusi repositori [shawroad/NLP_pytorch_project](https://github.com/shawroad/NLP_pytorch_project).
- Dalam laporan dan publikasi kampus, **cantumkan sumber kode** jika menggunakan atau memodifikasi skrip dari repo ini.
- Modifikasi untuk tugas harus dapat dijelaskan; hindari mengumpulkan hasil tanpa memahami alur `train.py` dan `model.py`.

---

## Referensi cepat

| Sumber | Tautan |
|--------|--------|
| Repositori asli | https://github.com/shawroad/NLP_pytorch_project |
| README asli (Mandarin) | [`README-old.md`](README-old.md) |
| Paper Transformer | Vaswani et al., *Attention Is All You Need* |
| Paper BERT | Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers* |

---

*Dokumen panduan hands-on — Program Studi Sains Data, Fakultas Sains, Institut Teknologi Sumatera.*
