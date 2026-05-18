"""
Ekstraksi kata kunci dengan TextRank (jieba.analyse).
Mata kuliah NLP Berbasis Transformer, Itera.
"""
import jieba.analyse

if __name__ == '__main__':
    with open('./news.txt', 'r', encoding='utf8') as f:
        text = f.read()

    keywords = jieba.analyse.textrank(
        text,
        topK=20,
        withWeight=True,
        allowPOS=('ns', 'n', 'vn', 'v', 'nr', 'nt'),
    )

    print('Kata kunci (TextRank), top 20:')
    for word, score in keywords:
        print(f'  {word}: {score:.4f}')
