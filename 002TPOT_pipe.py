#coding: utf-8

'''
TPOT
'''

from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score,KFold
from xgboost import XGBRegressor

# input data
data_input=pd.read_csv('D:/2022-Paper-LC-Alloy/20230423-alloy-HER-comp.csv', sep=',')
data_input2=pd.read_csv('D:/2022-Paper-LC-Alloy/thiswork.csv', sep=',')

labels=data_input['Overpotential (mV)']
features=data_input.drop('Overpotential (mV)', axis=1).drop('DOI', axis=1).drop('Composition', axis=1)
X_thiswork=data_input2.drop('Overpotential (mV)', axis=1).drop('Composition', axis=1)
y_thiswork=data_input2['Overpotential (mV)']

X_train,X_test,y_train,y_test=train_test_split(features, labels, test_size=0.2, random_state=42) #42

exported_pipeline = GradientBoostingRegressor(alpha=0.99, learning_rate=0.1, loss="ls", max_depth=10, max_features=0.9000000000000001, min_samples_leaf=1, min_samples_split=13, n_estimators=100, subsample=0.55)

# Fix random state in exported estimator
if hasattr(exported_pipeline, 'random_state'):
    setattr(exported_pipeline, 'random_state', 0)

exported_pipeline.fit(X_train, y_train)

feature_name=['Alloy_or_not', 'Pt_or_not', 'Weight_ratio', 'Particle_size', 'Surface_area', 'N_doped','pH','Current_density']
print([*zip(feature_name,exported_pipeline.feature_importances_)])
print(exported_pipeline.score(X_train, y_train))
print(exported_pipeline.score(X_test, y_test))

'''
strKFold = KFold(n_splits=5,shuffle=True,random_state=0)
scores = cross_val_score(exported_pipeline, features, labels, cv=strKFold)
print("KFold cross validation scores:{}".format(scores))
print("Mean score of KFold cross validation:{:.3f}".format(scores.mean()))
'''

y_pred = exported_pipeline.predict(X_test)
y_pred_2 = exported_pipeline.predict(X_train)
y_thiswork_pred= exported_pipeline.predict(X_thiswork)

result1=pd.DataFrame(columns=['y_test','y_pred'])
result2=pd.DataFrame(columns=['y_train','y_train_pred'])
result_thiswork=pd.DataFrame(columns=['y_thiswork','y_thiswork_pred'])

result1['y_test']=y_test
result1['y_pred']=y_pred
result2['y_train']=y_train
result2['y_train_pred']=y_pred_2
result_thiswork['y_thiswork']=y_thiswork
result_thiswork['y_thiswork_pred']=y_thiswork_pred


print("Train Accuracy r2: %.4g" % sk.metrics.r2_score(y_train, y_pred_2))
print("Test Accuracy r2: %.4g" % sk.metrics.r2_score(y_test, y_pred))
print("Train Accuracy MAE: %.4g" % sk.metrics.mean_absolute_error(y_train, y_pred_2))
print("Test Accuracy MAE: %.4g" % sk.metrics.mean_absolute_error(y_test, y_pred))
print("Train Accuracy mse: %.4g" % sk.metrics.mean_squared_error(y_train, y_pred_2))
print("Test Accuracy mse: %.4g" % sk.metrics.mean_squared_error(y_test, y_pred))

out_path1='D:/2022-Paper-LC-Alloy/result1.csv'
out_path2='D:/2022-Paper-LC-Alloy/result2.csv'
out_path3='D:/2022-Paper-LC-Alloy/thiswork_pred.csv'

result1.to_csv(out_path1,sep=',',index=False, header=False)
result2.to_csv(out_path2,sep=',',index=False, header=False)
result_thiswork.to_csv(out_path3,sep=',',index=False, header=False)

print('done')
