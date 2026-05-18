# Deployment — Bot Discord

Panduan menghubungkan **FastAPI backend** (`backend/`) ke **Discord** untuk demo live.

---

## Prasyarat

1. Checkpoint model ada di `Demo-day/models/` (lihat [`development/README.md`](../development/README.md)).
2. API berjalan dan `/health` mengembalikan `"status": "ok"`.
3. Akun Discord dengan izin membuat aplikasi bot.

---

## 1. Buat aplikasi bot di Discord

1. Buka [Discord Developer Portal](https://discord.com/developers/applications) → **New Application**.
2. Tab **Bot** → **Add Bot** → salin **Token** (jangan dibagikan publik).
3. Aktifkan **Message Content Intent** (Privileged Gateway Intents) jika bot harus membaca isi pesan di server.
4. Tab **OAuth2** → **URL Generator**:
   - Scopes: `bot`
   - Bot Permissions: `Send Messages`, `Read Message History` (minimal)
5. Buka URL undangan, tambahkan bot ke server uji.

---

## 2. Konfigurasi lingkungan

Salin `.env.example` ke `.env` di root `Demo-day/`:

```bash
cd Demo-day
cp .env.example .env
```

Isi minimal:

```env
DISCORD_TOKEN=your_bot_token_here
API_URL=http://127.0.0.1:8000
MODEL_TYPE=bert
```

| Variabel | Arti |
|----------|------|
| `DISCORD_TOKEN` | Token bot dari Developer Portal |
| `API_URL` | Base URL FastAPI (tanpa slash di akhir) |
| `BOT_PREFIX` | Opsional; default `!` — perintah `!chat` |
| `MENTION_ONLY` | `true` = bot hanya jawab saat di-mention |

---

## 3. Menjalankan (dua terminal)

**Terminal 1 — API**

```bash
cd Demo-day
source .venv/bin/activate
export MODEL_TYPE=bert
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

Uji:

```bash
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "WiFi kampus tidak bisa connect"}'
```

**Terminal 2 — Discord bot**

```bash
cd Demo-day
source .venv/bin/activate
python deployment/discord_bot/bot.py
```

Di channel Discord:

- `!chat WiFi tidak bisa connect`  
- atau mention bot: `@HelpdeskBot password email lupa`

---

## 4. Produksi ringan (opsional)

| Komponen | Saran demo kampus |
|----------|-------------------|
| API | `uvicorn` di VM/laptop lab; atau Docker + `gunicorn` + 1 worker |
| Bot | Proses systemd / `screen` / PM2 menjalankan `bot.py` |
| HTTPS | Jika API di cloud, bot di mesin lain set `API_URL=https://...` |

Pastikan firewall mengizinkan bot keluar ke Discord API dan ke host FastAPI.

---

## Troubleshooting

| Gejala | Solusi |
|--------|--------|
| `401 Unauthorized` (Discord) | Token salah atau token direset di portal |
| Bot online tapi tidak menjawab | Aktifkan **Message Content Intent**; cek `MENTION_ONLY` |
| `503` dari API | Model belum load — cek path checkpoint di `models/` |
| Jawaban kosong / aneh | Latih ulang dengan lebih banyak data; turunkan `beam_size` |
| `Connection refused` ke API | Jalankan uvicorn dulu; sesuaikan `API_URL` |

---

## Keamanan

- Jangan commit file `.env` atau token ke Git.
- Untuk demo publik, batasi bot ke satu server uji dengan role terbatas.
