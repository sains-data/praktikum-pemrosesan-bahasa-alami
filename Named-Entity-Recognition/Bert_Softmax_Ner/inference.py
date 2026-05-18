"""
Inferensi NER: BERT + Softmax per token.
"""
import torch
from transformers import BertTokenizer

from config import Config
from model import BertSoftmaxForNer

if __name__ == '__main__':
    input_sentence = 'Masukkan kalimat uji di sini'
    tokenizer = BertTokenizer.from_pretrained(Config.model_vocab_path)
    tokens = tokenizer.tokenize(input_sentence)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.LongTensor([input_ids])
    batch_masks = input_ids.gt(0)

    id2tag = {}
    with open('./data/msra/tags.txt', 'r') as f:
        for i, line in enumerate(f):
            id2tag[i] = line.strip()

    model = BertSoftmaxForNer().to(Config.device)
    model.load_state_dict(torch.load('./save_model/best_model.bin', map_location='cpu'))
    print('Model berhasil dimuat.')
    model.eval()

    logits = model(input_ids, token_type_ids=None, attention_mask=batch_masks, labels=None)
    logits = logits.squeeze(0)
    labels = torch.max(logits.data, 1)[1].cpu().numpy()
    tags = [id2tag[i] for i in labels]
    print('Token:', ' '.join(tokens))
    print('Tag  :', ' '.join(tags))
