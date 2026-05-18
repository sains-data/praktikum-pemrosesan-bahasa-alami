"""Engine inferensi Transformer — memakai kode Chatbot/transformer_chatbot."""
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

from backend.app.config import CHATBOT_ROOT, TRANSFORMER_CHECKPOINT, TRANSFORMER_VOCAB
from backend.app.engines.base import ChatEngine


class TransformerChatEngine(ChatEngine):
    name = "transformer"

    def __init__(
        self,
        checkpoint: Path | None = None,
        vocab_path: Path | None = None,
        device: str | None = None,
    ):
        self.checkpoint = Path(checkpoint or TRANSFORMER_CHECKPOINT)
        self.vocab_path = Path(vocab_path or TRANSFORMER_VOCAB)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._char2idx = None
        self._idx2char = None
        self._unk_id = 1

    def load(self) -> None:
        if not self.checkpoint.is_file():
            raise FileNotFoundError(
                f"Checkpoint Transformer tidak ditemukan: {self.checkpoint}\n"
                "Jalankan train.py + export.py di Chatbot/transformer_chatbot."
            )
        if not self.vocab_path.is_file():
            raise FileNotFoundError(
                f"vocab.pkl tidak ditemukan: {self.vocab_path}\n"
                "Salin dari transformer_chatbot/data/vocab.pkl setelah pre_process.py."
            )

        trans_dir = CHATBOT_ROOT / "transformer_chatbot"
        if str(trans_dir) not in sys.path:
            sys.path.insert(0, str(trans_dir))

        from transformer.transformer import Transformer

        with open(self.vocab_path, "rb") as f:
            data = pickle.load(f)
            self._idx2char = data["dict"]["idx2char"]
            self._char2idx = data["dict"]["char2idx"]
            self._unk_id = data["dict"].get("unk_id", 1)

        model = Transformer()
        model.load_state_dict(torch.load(self.checkpoint, map_location=self.device))
        model.to(self.device)
        model.eval()
        self._model = model

    def reply(self, message: str, **kwargs) -> str:
        if self._model is None:
            raise RuntimeError("Model belum dimuat.")

        text = message.strip()
        if not text:
            return "Silakan kirim pesan yang tidak kosong."

        sentence_in = [self._char2idx.get(c, self._unk_id) for c in list(text)]
        input_sent = torch.tensor(sentence_in, dtype=torch.long, device=self.device)
        input_length = torch.tensor([len(sentence_in)], dtype=torch.long, device=self.device)

        with torch.no_grad():
            hyps = self._model.recognize(
                input=input_sent,
                input_length=input_length,
                char_list=self._idx2char,
            )

        if not hyps:
            return ""

        out = hyps[0]["yseq"]
        chars = [self._idx2char[int(idx)] for idx in out]
        reply = "".join(chars)
        return reply.replace("<sos>", "").replace("<eos>", "").strip()
