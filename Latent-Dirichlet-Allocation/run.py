"""
Pemodelan topik LDA pada korpus teks (TF-IDF + sklearn).
Mata kuliah NLP Berbasis Transformer, Itera.

Keluaran:
  - top_vocab.csv : kata teratas per topik
  - result.csv    : probabilitas topik per dokumen
"""
import re

import jieba
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer


def top_words_data_frame(
    model: LatentDirichletAllocation,
    tf_idf_vectorizer: TfidfVectorizer,
    n_top_words: int,
) -> pd.DataFrame:
    """Kata dengan bobot tertinggi untuk setiap topik."""
    rows = []
    feature_names = tf_idf_vectorizer.get_feature_names()
    for topic in model.components_:
        top_indices = topic.argsort()[:-n_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_indices]
        rows.append(top_words)
    columns = [f'kata_{i + 1}' for i in range(n_top_words)]
    return pd.DataFrame(rows, columns=columns)


def predict_to_data_frame(model: LatentDirichletAllocation, X: np.ndarray) -> pd.DataFrame:
    """Distribusi probabilitas topik untuk setiap dokumen."""
    matrix = model.transform(X)
    columns = [f'P(topik {i + 1})' for i in range(len(model.components_))]
    return pd.DataFrame(matrix, columns=columns)


if __name__ == '__main__':
    # 1. Baca data dan tokenisasi
    document_column_name = '回答内容'  # sesuaikan jika pakai CSV Bahasa Indonesia
    pattern = (
        u'[\\s\\d,.<>/?:;\'\"[\\]{}()\\|~!\t"@#$%^&*\\-_=+a-zA-Z，。\\n'
        u'《》、？：；“”‘’｛｝【】（）…￥！—┄－]+'
    )
    df = (
        pd.read_csv('./answers.csv', encoding='utf-8-sig')
        .drop_duplicates()
        .rename(columns={document_column_name: 'text'})
    )
    df['cut'] = (
        df['text']
        .apply(lambda x: str(x))
        .apply(lambda x: re.sub(pattern, ' ', x))
        .apply(lambda x: ' '.join(jieba.lcut(x)))
    )
    print(f'Jumlah dokumen setelah deduplikasi: {len(df)}')

    # 2. TF-IDF
    tf_idf_vectorizer = TfidfVectorizer()
    tf_idf = tf_idf_vectorizer.fit_transform(df['cut'])
    print(f'Ukuran kosakata (fitur): {len(tf_idf_vectorizer.get_feature_names())}')

    # 3. LDA
    n_topics = 5
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=50,
        learning_method='online',
        learning_offset=50.0,
        random_state=0,
    )
    lda.fit(tf_idf)
    print(f'Pelatihan LDA selesai — jumlah topik: {n_topics}')

    n_top_words = 20
    top_words_df = top_words_data_frame(lda, tf_idf_vectorizer, n_top_words)
    top_words_df.index = [f'Topik {i + 1}' for i in range(n_topics)]
    top_words_df.to_csv('./top_vocab.csv', encoding='utf-8-sig')
    print('Disimpan: top_vocab.csv')

    X = tf_idf.toarray()
    predict_df = predict_to_data_frame(lda, X)
    predict_df.to_csv('./result.csv', encoding='utf-8-sig', index=False)
    print('Disimpan: result.csv')

    # Ringkasan topik dominan per dokumen
    dominant = predict_df.values.argmax(axis=1) + 1
    print('Contoh topik dominan (5 dokumen pertama):', dominant[:5].tolist())
