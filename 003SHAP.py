
#coding: utf-8

'''
SHAP
'''

# coding: utf-8

import csv
import math
import pandas as pd
import numpy as np
from numpy import *
import sklearn as sk
from sklearn import preprocessing, svm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import ShuffleSplit
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# plt.rcParams['figure.figsize'] = [8, 4]
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 12
my_cmap = plt.cm.get_cmap('RdBu').reversed()

tpot_data = pd.read_csv('.....csv', sep=',', dtype=np.float64)
features = tpot_data.drop('Overpotential (mV)', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['Overpotential (mV)'], random_state=42)

exported_pipeline = GradientBoostingRegressor(alpha=0.99, learning_rate=0.1, loss="ls", max_depth=10, max_features=0.9000000000000001, min_samples_leaf=1, min_samples_split=13, n_estimators=100, subsample=0.55)
# Fix random state in exported estimator
if hasattr(exported_pipeline, 'random_state'):
    setattr(exported_pipeline, 'random_state', 0)


exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
results_train = exported_pipeline.predict(training_features)

shap.initjs()
explainer = shap.Explainer(exported_pipeline)

y_base = explainer.expected_value
print(y_base)

predictt = exported_pipeline.predict(training_features)
print(predictt.mean())

shap_values = explainer.shap_values(features)
np.savetxt(".....csv", shap_values, delimiter=",")

fig = plt.figure() #figsize=(8, 4)
shap.summary_plot(shap_values, features, show=False, cmap=my_cmap)

plt.show()



