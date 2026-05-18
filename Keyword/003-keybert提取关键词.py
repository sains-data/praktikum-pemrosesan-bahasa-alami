"""
Ekstraksi kata kunci dengan KeyBERT (embedding BERT).
Mata kuliah NLP Berbasis Transformer, Itera.

Instalasi: pip install keybert jieba sentence-transformers
"""
from keybert import KeyBERT
import jieba

if __name__ == '__main__':
    # Skor = cosine similarity embedding dokumen vs embedding kandidat kata/frasa
    # Untuk Indonesia, pertimbangkan: paraphrase-multilingual-MiniLM-L12-v2
    model = KeyBERT('bert-base-chinese')

    with open('./news.txt', 'r', encoding='utf8') as f:
        text = f.read()

    doc = ' '.join(jieba.cut(text))
    keywords = model.extract_keywords(
        doc,
        keyphrase_ngram_range=(1, 2),
        top_n=20,
    )

    print('Kata kunci (KeyBERT), top 20:')
    for phrase, score in keywords:
        print(f'  {phrase}: {score:.4f}')
