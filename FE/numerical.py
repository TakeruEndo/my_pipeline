import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn.preprocessing import PolynomialFeatures


def standard_scaler(df, target):
    """標準化
    """
    X = df.drop(target, axis=1)
    y = df[target]
    scaler = StandardScaler()
    scaler.fit(X)
    new_df = scaler.transform(X)
    print(new_df)
    new_df[target] = y
    return new_df


def min_max_scaler(df):
    """Min=maxスケーリング
    """
    scaler = MinMaxScaler()
    scaler.fit(df)
    df = scaler.transform(df)
    return df


def trans_log1p(df, target):
    """log(x+1)
    逆の処理
    np.expm1(x)
    """
    df[target] = np.log1p(df[target].values)
    return df


def cliping(df, start, end):
    """クリッピング
    外れ値の助教に使う
    """
    start_point = df.quantile(start)
    end_point = df.quantile(end)
    df = df.clip(start_point, end_point, axis=1)
    return df


def binning(x, num_bins):
    """数値を区間に分けてカテゴリ変数として使う
    """
    binned = pd.cut(x, num_bins, label=False)
    return binned


def trans_ranking(x):
    """ランキングに変換
    """
    rank = pd.Series(x).rank()
    return rank.values


def trans_rank_gauss(df):
    """順位変換後正規分布に変換する
    ニューラルネットでいい性能を示す
    """
    transformer = QuantileTransformer(
        n_quantiles=100, random_state=0, output_distribution='normal')
    transformer.fit(df)
    df = transformer.transform(df)
    return df


def shift(df, target, shift_size):
    df['{}_shift_{}'.format(target, shift_size)] = df[target].shift(shift_size)
    return df


def move_average(df, target, shift_size, window_size):
    """移動平均
    """
    df['{}_shift{}_window{}_mean'.format(target, shift_size, window_size)] = df[target].shift(
        shift_size).rolling(window=window_size).mean()
    df['{}_shift{}_window{}_max'.format(target, shift_size, window_size)] = df[target].shift(
        shift_size).rolling(window=window_size).max()


def simple_statitics(df, id, target):
    """単純な統計量をとる
    ・合計
    ・平均
    ・割合
    ・最大
    ・最小
    https://deepage.net/features/pandas-groupby.html
    """
    df['max_' + target + '_every_' + id] = df.groupby([id])[target].max()
    df['max_' + target + '_every_' + id] = df.groupby([id])[target].min()
    df['max_' + target + '_every_' + id] = df.groupby([id])[target].std()
    df['max_' + target + '_every_' + id] = df.groupby([id])[target].mean()
    return df


def get_power(df, target):
    """2乗を返す
    """
    df[target + '**2'] = df[target] * df[target]
    return df


def rolling_features(df, target):
    window_sizes = [6, 12]
    for window in window_sizes:
        df["rolling_mean_" + str(window) + target
           ] = df[target].rolling(window=window).mean()
        df["rolling_std_" + str(window) + target
           ] = df[target].rolling(window=window).std()
        df["rolling_min_" + str(window) + target
           ] = df[target].rolling(window=window).min()
        df["rolling_max_" + str(window) + target
           ] = df[target].rolling(window=window).max()


def get_polynomialFeatures(df, columns):
    df_importance = df[columns]
    pf = PolynomialFeatures(degree=2, include_bias=False)
    new_X = pf.fit_transform(df_importance)
    for i in range(new_X.shape[1]):
        df[str(i) + '_new_col'] = new_X[:, i]
    return df
