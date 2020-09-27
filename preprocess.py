import sys
import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures

from FE.categorical import *
from FE.numerical import *
# from FE.missing import *
from FE.datetime import *
from FE.nlp.embeddings import *
from FE.nlp.nlp import *


def preprocess(train, test, targeti, model_name):

    train, test = target_encoding(train, test, columns, target)

    df = pd.concat([train, test], axis=0)

    num_col_1 = []
    cat_column_1 = []

    return df
