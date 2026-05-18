# Data Demo-day — Helpdesk Teknis

## Format BERT (`chatbot_bert.txt`)

Satu baris per dialog, pemisah `=`:

```text
WiFi kampus tidak bisa connect?=Coba lupa jaringan lalu sambung ulang ke SSID kampus dengan akun SSO.
```

## Format Transformer (`dialog_transformer.txt`)

Pemisah `|`:

```text
Bagaimana cek versi kernel?|Jalankan perintah uname -r di terminal.
```

## Sumber

- **Produksi:** [Ubuntu Dialogue Corpus](https://github.com/rkadlec/ubuntu-ranking-dataset-creator)  
- **Prototipe:** `sample_helpdesk_id.txt` (bahasa Indonesia, tema troubleshooting)
