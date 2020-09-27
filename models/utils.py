import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


def save_feature_impotances(columns, fe_imp, output_path):
    feature_importances = pd.DataFrame()
    feature_importances['feature'] = columns
    feature_importances['importance'] = fe_imp
    feature_importances = feature_importances.sort_values(
        by='importance', ascending=False)
    feature_importances.to_csv(os.path.join(output_path, 'fe_imp.csv'))
    fig = plt.figure(figsize=(16, 16))
    sns.barplot(data=feature_importances, x='importance', y='feature')
    fig.savefig(output_path + '/fe_imp.png')


def save_params(params, output_path):
    with open(os.path.join(output_path, 'params.txt'), mode='w') as f:
        for k, v in params.items():
            f.write("{}: {}".format(k, v))
            f.write('\n')
