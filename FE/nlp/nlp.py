import pandas as pd
import json
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def get_ngram(df, target):
    """nグラム
    `fit()`
    データの統計値を内部メモリに保存する
    `transform()`
    fit()の統計量を使って、データを書き換える
    `fit_transform()`
    fit()＋transform()を行う
    """
    bow_converter = CountVectorizer(token_pattern='(?u)\\b\\w+\\b')
    bigram_converter = CountVectorizer(ngram_range=(2, 2),
                                       token_pattern='(?u)\\b\\w+\\b')
    trigram_converter = CountVectorizer(ngram_range=(3, 3),
                                        token_pattern='(?u)\\b\\w+\\b')

    # 変換器を適用して語彙数を確認する
    words_list = bow_converter.fit＿transform(df[target])
    words = bow_converter.get_feature_names()
    words_df = pd.DataFrame(
        words_list.toarray(), columns=words)

    bigram_list = bigram_converter.fit＿transform(df[target])
    bigrams = bigram_converter.get_feature_names()
    bigrams_df = pd.DataFrame(
        bigram_list.toarray(), columns=bigrams)

    trigram_list = trigram_converter.fit＿transform(df[target])
    trigrams = trigram_converter.get_feature_names()
    trigram_df = pd.DataFrame(
        trigram_list.toarray(), columns=trigrams)

    print(len(words), len(bigrams), len(trigrams))
    print(bigrams[:10])

    # TODO
    """単語除去
    - 頻度による単語除去
    - ストップワードによる単語除去
    """
    use_cols = []
    for col in words_df.columns:
        if words_df.shape[0]*0.0025 < words_df[col].sum():
            use_cols.append(col)
    words_df = words_df[use_cols]

    use_cols = []
    for col in bigrams_df.columns:
        if bigrams_df.shape[0]*0.0025 < bigrams_df[col].sum():
            use_cols.append(col)
    bigrams_df = bigrams_df[use_cols]

    use_cols = []
    for col in trigram_df.columns:
        if trigram_df.shape[0]*0.0025 < trigram_df[col].sum():
            use_cols.append(col)
    trigram_df = trigram_df[use_cols]

    words_cols = {col: col + '_words' for col in words_df.columns}
    words_df = words_df.rename(columns=words_cols)
    bigram_cols = {col: col + '_bigrams' for col in bigrams_df.columns}
    bigrams_df = bigrams_df.rename(columns=bigram_cols)
    trigram_cols = {col: col + '_words' for col in trigram_df.columns}
    trigram_df = trigram_df.rename(columns=trigram_cols)

    data = pd.concat([df.reset_index(drop=True), words_df,
                      bigrams_df, trigram_df], axis=1)

    return data


def get_tf_idf(df, target):
    """tf-idf
    意味のある単語が強調されるように特徴を表現する

    bow(w,d) = [文書dないの単語wの出現回数]
    tf(w,d) = bow(w,d)/[文書d内の単語数]
    idf(w) =[全文書数N]/[単語wが含まれる文書数]
    tf-idf(w, d) = tf(w,d)*idf(w,d)
    """
    tfidf = TfidfVectorizer()
    df_ = tfidf.fit_transform(df[target])
    tfidf_df = pd.DataFrame(df_.toarray(), columns=tfidf.get_feature_names())

    # 使うカラムを決める
    use_cols = []
    thld_q90 = np.percentile(tfidf_df.std().values, 90)
    for col in tfidf_df.columns:
        if thld_q90 < tfidf_df[col].std():
            use_cols.append(col)
    tfidf_df = tfidf_df[use_cols]

    tfidf_cols = {col: col + '_tfidf' for col in tfidf_df.columns}
    tfidf_df = tfidf_df.rename(columns=tfidf_cols)

    data = pd.concat([df.reset_index(drop=True), tfidf_df], axis=1)

    return data


def special_words(df, target, word):
    df['{}'.format(word)] = df[target].map(lambda x: 1 if 'Ｒ' in str(x) else 0)
    return df

