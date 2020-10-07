import sys
import re
import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures

from FE.categorical import *
from FE.numerical import *
from FE.missing import *
from FE.datetime import *
from FE.nlp.embeddings import *
from FE.nlp.nlp import *


def preprocess(train, test, target, model_name):

    df = pd.concat([train, test], axis=0)

    cat_cols = []

    binary_cols = []

    num_cols = []

    delete_cols = []

    """datetimeの作成
    """
    df = get_date_time(df, 'imp_at')

    """category変数のエンコーディング
    """
    df = frequency_encoding(df, cat_cols)
    df = label_encoding(df, binary_cols)

    statistics_cols = []
    df = group_statitics(df, ['目的id'], statistics_cols)

    for i, j in zip(df.isnull().sum(), df.isnull().sum().index):
        print(i, j)

    return df
