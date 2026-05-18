"""
Ekstraksi kata kunci dengan TF-IDF (jieba.analyse).
Mata kuliah NLP Berbasis Transformer, Itera.
"""
import jieba.analyse

if __name__ == '__main__':
    with open('./news.txt', 'r', encoding='utf8') as f:
        text = f.read()

    # jieba menghitung TF-IDF pada satu dokumen; filter POS untuk kata bermakna
    # ns=tempat, n=kata benda, vn=verba nominalisasi, v=verba, nr=nama orang, nt=nama organisasi
    keywords = jieba.analyse.extract_tags(
        text,
        topK=20,
        withWeight=True,
        allowPOS=('ns', 'n', 'vn', 'v', 'nr', 'nt'),
    )

    print('Kata kunci (TF-IDF), top 20:')
    for word, score in keywords:
        print(f'  {word}: {score:.4f}')
