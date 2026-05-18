"""Engine inferensi BERT — memakai kode Chatbot/Bert_chatbot."""
import os
import sys
from pathlib import Path

import torch

from backend.app.config import BERT_CHECKPOINT, CHATBOT_ROOT
from backend.app.engines.base import ChatEngine


class BertChatEngine(ChatEngine):
    name = "bert"

    def __init__(self, checkpoint: Path | None = None, device: str | None = None):
        self.checkpoint = Path(checkpoint or BERT_CHECKPOINT)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None

    def load(self) -> None:
        if not self.checkpoint.is_file():
            raise FileNotFoundError(
                f"Checkpoint BERT tidak ditemukan: {self.checkpoint}\n"
                "Latih di Chatbot/Bert_chatbot lalu salin bert_dream.bin ke Demo-day/models/"
            )

        bert_dir = CHATBOT_ROOT / "Bert_chatbot"
        if not bert_dir.is_dir():
            raise FileNotFoundError(f"Folder Bert_chatbot tidak ditemukan: {bert_dir}")
        if str(bert_dir) not in sys.path:
            sys.path.insert(0, str(bert_dir))

        from bert_model import BertConfig
        from seq2seq_bert import Seq2SeqModel
        from tokenizer import load_bert_vocab

        prev_cwd = os.getcwd()
        os.chdir(bert_dir)
        try:
            word2idx = load_bert_vocab()
            config = BertConfig(len(word2idx))
            model = Seq2SeqModel(config)
            state = torch.load(self.checkpoint, map_location=self.device)
            model.load_state_dict(state)
            model.eval()
            self._model = model
        finally:
            os.chdir(prev_cwd)

    def reply(self, message: str, beam_size: int = 3, **kwargs) -> str:
        if self._model is None:
            raise RuntimeError("Model belum dimuat. Panggil load() terlebih dahulu.")
        text = message.strip()
        if not text:
            return "Silakan kirim pesan yang tidak kosong."
        return self._model.generate(text, beam_size=beam_size, device=self.device)
