import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sklearn
import scipy

from sklearn import metrics, model_selection, preprocessing, ensemble, tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler


from xgboost import XGBClassifier
xgb_cl = XGBClassifier()

heart = pd.read_csv('//Users//behroozkeshavarzi//FinalHeart.csv')
heart.drop(['Unnamed: 0'],axis=1,inplace=True)

X = heart.drop(['num'],axis=1).copy()
Y = heart.num.copy()
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size= 0.33, stratify=Y)

SC = StandardScaler()
X_scaled = SC.fit_transform(X)

xgb_cl = XGBClassifier()
xgb_cl.set_params(random_state=144)
xgb_cl.fit(X_train,Y_train)

R2_train = np.round(xgb_cl.score(X_train,Y_train),3)
R2_test  = np.round(xgb_cl.score(X_test,Y_test),3)

print('Training score is :'+ str(R2_train))
print('Testing  score is :'+ str(R2_test))

xgb_cl = XGBClassifier(gamma = 4, max_depth = 10)
xgb_cl.set_params(random_state=144)
xgb_cl.fit(X_train,Y_train)

R2_train = np.round(xgb_cl.score(X_train,Y_train),3)
R2_test  = np.round(xgb_cl.score(X_test,Y_test),3)

print('Training score is :'+ str(R2_train))
print('Testing  score is :'+ str(R2_test))


Y_0 = Y_train==0
Y_1 = Y_train==1
Y_2 = Y_train==2
Y_3 = Y_train==3
Y_4 = Y_train==4

prob_train = xgb_cl.predict_proba(X_train)
prob_test  = xgb_cl.predict_proba(X_test)

from sklearn.metrics import confusion_matrix

conf_data_train = confusion_matrix(Y_train,xgb_cl.predict(X_train))
conf_data_test  = confusion_matrix(Y_test,xgb_cl.predict(X_test))

fig, ax = plt.subplots(1,2,figsize= (18,6))
sns.heatmap(conf_data_train, ax = ax[0], annot=True)
sns.heatmap(conf_data_test,  ax = ax[1], annot=True)

ax[0].tick_params(axis='both',labelsize=16)
ax[0].set_xlabel('Measured', fontsize= 18)
ax[0].set_ylabel('Predicted', fontsize= 18)
ax[0].set_title('Train Set',fontsize= 22)

ax[1].tick_params(axis='both',labelsize=16)
ax[1].set_xlabel('Measured', fontsize= 18)
ax[1].set_ylabel('Predicted', fontsize= 18)
ax[1].set_title('Test Set',fontsize= 22);

conf_train_norm = np.zeros_like(conf_data_train, dtype=float)
conf_test_norm = np.zeros_like(conf_data_test, dtype=float)

sum_vec_train = np.sum(conf_data_train, axis=1)
sum_vec_test = np.sum(conf_data_test, axis=1)

print('*' * 100)

for irow in range(conf_train_norm.shape[0]):
    conf_train_norm[irow, :] = 100 * np.round(conf_data_train[irow, :] / sum_vec_train[irow], 2)
    conf_test_norm[irow, :] = 100 * np.round(conf_data_test[irow, :] / sum_vec_test[irow], 2)

fig, ax = plt.subplots(1, 2, figsize=(18, 6))
sns.heatmap(conf_train_norm, ax=ax[0], annot=True)
sns.heatmap(conf_test_norm, ax=ax[1], annot=True)

ax[0].tick_params(axis='both', labelsize=16)
ax[0].set_xlabel('Measured', fontsize=18)
ax[0].set_ylabel('Predicted', fontsize=18)
ax[0].set_title('Train Set (Percent)', fontsize=22)

ax[1].tick_params(axis='both', labelsize=16)
ax[1].set_xlabel('Measured', fontsize=18)
ax[1].set_ylabel('Predicted', fontsize=18)
ax[1].set_title('Test Set (Percent)', fontsize=22)

plt.show()
