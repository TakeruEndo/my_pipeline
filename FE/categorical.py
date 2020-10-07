import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import KFold


def one_hot_encoding(df, cat_cols):
    """ワンホット表現
    """
    ohe = OneHotEncoder(sparse=False, categories='auto')
    for c in cat_cols:
        df[c] = df[c].fillna('missing')
    ohe.fit(df[cat_cols])

    # ダミー変数の列名の作成
    columns = []
    for i, c in enumerate(cat_cols):
        columns += [f'{c}_{v}' for v in ohe.categories_[i]]

    # ダミー変数をデータフレームに変換
    dummy_vals = pd.DataFrame(ohe.transform(df[cat_cols]), columns=columns)
    dummy_vals = dummy_vals.reset_index(drop=True)
    df = df.reset_index(drop=True)
    df = pd.concat([df, dummy_vals], axis=1)
    return df


def label_encoding(df, cat_cols):
    """
    しっかり最後にカラムを消すこと
    """
    for c in cat_cols:
        values = df[c].copy().fillna('missing')
        le = LabelEncoder()
        le.fit(values)
        df[c + '_label'] = le.transform(values)
        # df[c] = le.transform(values)
    return df


def frequency_encoding(df, cat_cols):
    for c in cat_cols:
        values = df[c].copy().fillna('missing')
        freq = values.value_counts()
        # カテゴリの出現回数で置換
        df[c + '_freq'] = values.map(freq)
    return df


def target_encoding(train, test, cat_cols, target):
    # 変数をループしてtarget encoding
    for c in cat_cols:
        train[c] = train[c].fillna('missing')
        test[c] = test[c].fillna('missing')
        # 学習データ全体で各カテゴリにおけるtargetの平均を計算
        data_tmp = pd.DataFrame({c: train[c], 'target': train[target]})
        target_mean = data_tmp.groupby(c)['target'].mean()
        # テストデータのカテゴリを置換
        test[c + '_target'] = test[c].map(target_mean)

        # 学習データの変換後の値を格納する配列を準備
        tmp = np.repeat(np.nan, train.shape[0])

        # 学習データを分割
        kf = KFold(n_splits=4, shuffle=True, random_state=71)
        for idx_1, idx_2 in kf.split(train):
            # out-of-foldで各カテゴリにおける目的変数の平均を計算
            target_mean = data_tmp.iloc[idx_1].groupby(c)['target'].mean()
            # 変換後の値を一時に配列を格納
            tmp[idx_2] = train[c].iloc[idx_2].map(target_mean)

        # 変換後のデータで元の変数を置換
        train[c + '_target'] = tmp

    return train, test
