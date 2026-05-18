"""
Konversi pasangan dialog Ubuntu (TSV/CSV) ke format Chatbot proyek.

Contoh input TSV (header opsional):
  question\tanswer
  How do I update?\tRun apt update...

Usage:
  python scripts/convert_ubuntu_to_chatbot.py --input pairs.tsv --output_dir ./data
"""
import argparse
import csv
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Konversi UDC ke format BERT/Transformer")
    p.add_argument("--input", required=True, help="File TSV/CSV: kolom 1=pertanyaan, 2=jawaban")
    p.add_argument("--output_dir", default="./data", help="Folder keluaran")
    p.add_argument("--delimiter", default="\t", help="Pemisah kolom")
    p.add_argument("--skip_header", action="store_true", help="Baris pertama adalah header")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = []
    with open(args.input, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=args.delimiter)
        for i, row in enumerate(reader):
            if args.skip_header and i == 0:
                continue
            if len(row) < 2:
                continue
            q, a = row[0].strip(), row[1].strip()
            if q and a:
                pairs.append((q, a))

    bert_path = out_dir / "chatbot_bert.txt"
    trans_path = out_dir / "dialog_transformer.txt"
    bert_path.write_text(
        "\n".join(f"{q}={a}" for q, a in pairs) + "\n", encoding="utf-8"
    )
    trans_path.write_text(
        "\n".join(f"{q}|{a}" for q, a in pairs) + "\n", encoding="utf-8"
    )
    print(f"OK: {len(pairs)} pasangan → {bert_path.name}, {trans_path.name}")


if __name__ == "__main__":
    main()
