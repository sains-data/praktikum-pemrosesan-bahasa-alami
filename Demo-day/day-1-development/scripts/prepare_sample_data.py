"""
Buat data sampel helpdesk (ID) untuk demo BERT & Transformer.
Jalankan dari folder day-1-development/: python3 scripts/prepare_sample_data.py
"""
from pathlib import Path

SAMPLE_PAIRS = [
    ("WiFi kampus tidak bisa connect?", "Coba lupakan jaringan WiFi kampus, lalu sambung ulang dan login dengan akun SSO."),
    ("Lupa password portal mahasiswa?", "Gunakan fitur reset password di halaman login portal, lalu cek email kampus."),
    ("Cara update sistem Ubuntu?", "Jalankan sudo apt update && sudo apt upgrade -y, lalu restart jika diminta."),
    ("Layar hitam setelah login?", "Tekan Ctrl+Alt+F3, login, lalu jalankan sudo apt install --reinstall ubuntu-desktop."),
    ("Tidak bisa install paket apt?", "Periksa koneksi internet dan jalankan sudo apt update terlebih dahulu."),
    ("Bagaimana cek versi kernel?", "Ketik uname -r di terminal untuk melihat versi kernel."),
    ("Printer lab tidak terdeteksi?", "Pastikan kabel USB terpasang dan layanan CUPS berjalan: sudo systemctl status cups."),
    ("Email kampus penuh?", "Kosongkan folder Spam dan hapus lampiran besar di kotak masuk."),
    ("VPN kampus error?", "Pastikan client VPN versi terbaru dan gunakan kredensial SSO yang sama."),
    ("Moodle lambat dibuka?", "Coba browser lain, bersihkan cache, atau akses di jam sibuk rendah."),
    ("Git push ditolak permission denied?", "Periksa SSH key terdaftar di GitLab dan remote URL memakai git@ bukan https salah."),
    ("Python pip install gagal?", "Gunakan python3 -m pip install --user nama_paket atau virtualenv."),
    ("Docker permission denied?", "Tambahkan user ke grup docker: sudo usermod -aG docker $USER lalu logout."),
    ("Disk hampir penuh?", "Jalankan du -sh /* untuk cek folder besar, lalu apt autoremove dan apt clean."),
    ("Laptop panas saat training model?", "Kurangi batch size, gunakan GPU cloud, atau aktifkan throttle monitoring."),
]

BERT_SEP = "="
TRANS_SEP = "|"


def main():
    base = Path(__file__).resolve().parents[1] / "data"
    base.mkdir(parents=True, exist_ok=True)

    bert_path = base / "chatbot_bert.txt"
    trans_path = base / "dialog_transformer.txt"
    sample_path = base / "sample_helpdesk_id.txt"

    lines_bert = []
    lines_trans = []
    for q, a in SAMPLE_PAIRS:
        q, a = q.strip(), a.strip()
        lines_bert.append(f"{q}{BERT_SEP}{a}")
        lines_trans.append(f"{q}{TRANS_SEP}{a}")

    bert_path.write_text("\n".join(lines_bert) + "\n", encoding="utf-8")
    trans_path.write_text("\n".join(lines_trans) + "\n", encoding="utf-8")
    sample_path.write_text("\n".join(f"{q}\t{a}" for q, a in SAMPLE_PAIRS) + "\n", encoding="utf-8")

    print(f"Disimpan: {bert_path} ({len(SAMPLE_PAIRS)} pasangan)")
    print(f"Disimpan: {trans_path}")
    print(f"Disimpan: {sample_path}")
    print("\nSalin ke Chatbot/Bert_chatbot/data/chatbot.txt untuk mulai latih BERT.")


if __name__ == "__main__":
    main()
