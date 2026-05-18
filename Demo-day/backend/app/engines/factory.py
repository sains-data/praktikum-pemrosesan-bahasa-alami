from backend.app import config
from backend.app.engines.bert_engine import BertChatEngine
from backend.app.engines.transformer_engine import TransformerChatEngine


def build_engine():
    if config.MODEL_TYPE == "transformer":
        return TransformerChatEngine()
    if config.MODEL_TYPE == "bert":
        return BertChatEngine()
    raise ValueError(f"MODEL_TYPE tidak dikenal: {config.MODEL_TYPE}. Gunakan bert atau transformer.")
