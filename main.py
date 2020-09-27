import sys
import os
import copy
import logging
from time import time
from datetime import datetime

import pandas as pd
import numpy as np

from tqdm.notebook import trange


from preprocess import preprocess
from utils import select_model
from models import lightGBM


def submit_csv(pred):
    # 提出用に整形
    submit_df = pd.DataFrame({target: pred})
    submit_df.to_csv(output_path + '/submission_{}_{}-{}-{}-{}.csv'.format(model_name, str(
        dt_now.year), str(dt_now.day), str(dt_now.hour), str(dt_now.minute)), index=False)


def submit_csv_dict(pred_dict):
    # 提出用に整形
    for k, v in pred_dict.items():
        submit_df = pd.DataFrame({target: v})
        submit_df.to_csv(output_path + '/{}_submission_{}_{}-{}-{}-{}.csv'.format(k, model_name, str(
            dt_now.year), str(dt_now.day), str(dt_now.hour), str(dt_now.minute)), index=False)


if __name__ == '__main__':

    dt_now = datetime.now()
    model_name = 'lgbm'
    target = 'target'
    output_path = 'outputs/{}_{}-{}-{}-{}'.format(model_name, str(
        dt_now.year), str(dt_now.day), str(dt_now.hour), str(dt_now.minute))
    os.mkdir(output_path)

    logging.basicConfig(filename=os.path.join(
        output_path, 'logger.log'), level=logging.INFO)

    ROOT = "inputs"

    train = pd.read_csv(f"{ROOT}/train.csv")
    test = pd.read_csv(f"{ROOT}/test.csv")
    """重複行を削除
    keep: False <= 重複した行を全て削除 (defaultはTrue)
    inplace: True <= 元のデータセットから削除
    subset: 行を選択 (subset=['Patient', 'Weeks'])
    """
    train.drop_duplicates(inplace=True)

    """特徴量生成
    """
    data = preprocess(train, test, target, model_name)

    print('columns_length: ', len(data.columns))
    train = data[: len(train)]
    test = data[len(train):]

    if False:
        train.to_csv('inputs/train2.csv', index=False)
        test.to_csv('inputs/test2.csv', index=False)
        sys.exit()

    """モデルとパラメータの設定
    """
    # TODO: txtファイルで重み保存フォルダに一緒に保存

    """学習と推論
    """
    train_x = train.drop(target, axis=1)
    y = train[target]
    test = test.drop(target, axis=1)

    model = select_model(model_name, train_x, y, test, output_path, 'kfold', 5)
    pred = model.trainer()

    """提出
    """
    if model_name == 'linear':
        submit_csv_dict(pred)
    else:
        # 出力フォルダの作成
        submit_csv(pred)