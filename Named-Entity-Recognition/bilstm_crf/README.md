# Hands-On: BiLSTM + CRF untuk NER

**Latih:** [`../BiLSTM_CRF_Ner/train.py`](../BiLSTM_CRF_Ner/train.py)  
**Pra-pemrosesan:** [`../BiLSTM_CRF_Ner/data/data_process.py`](../BiLSTM_CRF_Ner/data/data_process.py)

---

## 1. Teori

**Embedding** kata → **BiLSTM** menangkap konteks dua arah → **CRF** mendekode urutan tag optimal (Viterbi).

Baseline klasik sebelum era transformer; tetap relevan untuk memahami **structured prediction** pada NER.

---

## 2. Data

- `data/renmin.txt` — korpus People Daily (format `kata/tag`)  
- Pipeline: `data_process.py` → `renmindata.pkl` (jalankan fungsi `data2pkl()`)

Tag contoh: `B-ns` (lokasi), `B-nr` (orang), `B-nt` (organisasi).

---

## 3. Langkah hands-on

```bash
cd NER/BiLSTM_CRF_Ner/data
python data_process.py   # hasilkan renmindata.pkl

cd ..
python train.py
```

Mode interaktif prediksi (di `train.py`, bagian bawah): masukkan teks Mandarin.

Konfigurasi: [`config.py`](../BiLSTM_CRF_Ner/config.py).

---

## 4. Metrik

`train.py` memakai `sklearn.metrics`: **precision, recall, F1** dan `classification_report` pada level token/tag.

Untuk evaluasi entitas, konversi ke span dengan `utils.format_result`.

---

## 5. Deploy

- Model disimpan di `save_model/` setelah pelatihan  
- Ringan untuk CPU; cocok sebagai baseline sebelum BERT  

---

## Referensi

- [shawroad/NLP_pytorch_project](https://github.com/shawroad/NLP_pytorch_project)
