# encoding: utf-8
import os

import flask
import torch
from flask import Flask, render_template, request
from loguru import logger
from peft import PeftModel
from modeling_chatglm import ChatGLMForConditionalGeneration
from tokenization_chatglm import ChatGLMTokenizer

logger.add('./logger/log_rizhi.log')

app = Flask("ChatGLM-Demo")

# Sesuaikan path model dasar dan adaptor LoRA
CHATGLM_BASE = os.environ.get('CHATGLM_PATH', '/path/to/chatglm_pretrain')
LORA_PATH = os.environ.get('LORA_PATH', './output/global_step-2000')


@app.route('/')
def index():
    return render_template('chatgpt_clone.html')


@app.route('/chatgpt-clone', methods=['POST', 'GET'])
def chatgpt_clone():
    question = request.args.get('question', '')

    logger.info('Pertanyaan: {}'.format(question))
    question = str(question).strip()
    if len(question) > 0:
        def stream():
            last_answer = ''
            for s in gen_answer(question):
                if s == 'stop':
                    logger.info('Jawaban: {}'.format(last_answer))
                    data = '[DONE]'
                else:
                    ids_list = s.tolist()[0]
                    data = tokenizer.decode(ids_list).replace('<eop>', '')
                    last_answer = data
                yield "data: %s\n\n" % data.replace('\n', '<br />').replace(question, '')
        return flask.Response(stream(), mimetype="text/event-stream")
    return 'Pertanyaan kosong'


model = ChatGLMForConditionalGeneration.from_pretrained(CHATGLM_BASE)
tokenizer = ChatGLMTokenizer.from_pretrained(CHATGLM_BASE)
model = PeftModel.from_pretrained(model, LORA_PATH, torch_dtype=torch.float32)
if torch.cuda.is_available():
    model.half().cuda()


def gen_answer(question):
    max_len = 512
    max_src_len = 128
    max_tgt_len = max_len - max_src_len - 3
    src_tokens = tokenizer.tokenize(question)
    if len(src_tokens) > max_src_len:
        src_tokens = src_tokens[:max_src_len]
    tokens = src_tokens + ['[gMASK]', '<sop>']
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.tensor([input_ids])
    generation_kwargs = {
        "min_length": 5,
        "max_new_tokens": max_tgt_len,
        "top_p": 0.7,
        "temperature": 0.95,
        "do_sample": False,
        "num_return_sequences": 1,
    }
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
    return model.stream_generate(input_ids, **generation_kwargs)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6006, debug=True)
