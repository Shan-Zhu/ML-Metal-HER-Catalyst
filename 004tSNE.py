
#coding: utf-8

'''
SHAP
'''

import shap


# coding: utf-8

import csv
import math
import pandas as pd
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import seaborn as sns
from matplotlib.colors import ListedColormap


tpot_data = pd.read_csv('D:/2022-Paper-LC-Alloy/20230423-alloy-HER-comp-noDOI.csv', sep=',', dtype=np.float64)
features = tpot_data.drop('Overpotential (mV)', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['Overpotential (mV)'], random_state=1)


tsne = TSNE(n_components=2,perplexity=5)
X_tsne = tsne.fit_transform(features)
X_tsne_data = np.vstack((X_tsne.T, tpot_data['Overpotential (mV)'])).T
df_tsne = pd.DataFrame(X_tsne_data, columns=['Dim1', 'Dim2', 'label'])
df_tsne.head()

df_tsne.to_csv('tsne-data.csv', index=False)

# plt.figure(figsize=(8, 8))
# sns.scatterplot(data=df_tsne, hue='class', x='Dim1', y='Dim2',palette='vlag')
# # plt.savefig('my_plot.tiff', dpi=600, format='tiff')
# plt.show()
