
#coding: utf-8

'''
TPOT
'''

import pandas as pd
from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split


# input data
data_input=pd.read_csv('....csv', sep=',')

labels=data_input['Overpotential (mV)']
features=data_input.drop('Overpotential (mV)', axis=1).drop('DOI', axis=1).drop('Composition', axis=1)

X_train,X_test,y_train,y_test=train_test_split(features, labels, test_size=0.2, random_state=42)

tpot = TPOTRegressor(generations=100, random_state=0,verbosity=2,template='Regressor')

tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('....py')
