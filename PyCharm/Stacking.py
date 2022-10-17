#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sklearn
import scipy

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import StackingClassifier

from PlottingFunction import lda_1Dplot, plotModel, data_1Dplot
from Classification import model_fit, plot_confusion_matrix, model_fit2

from sklearn.preprocessing import StandardScaler

glm = LogisticRegression(max_iter = 1e5)

rfc = RandomForestClassifier(criterion='gini', max_depth = 3, max_features = 8, min_samples_split = 10,
                             random_state = 144,min_weight_fraction_leaf = 0.01)

svc = svm.SVC(C = 0.05, kernel = 'poly', max_iter = 1e5, degree = 1, probability = True, random_state = 144, gamma = 0.05)

gbc = GradientBoostingClassifier(random_state = 144, n_estimators = 5000, max_features = 5, max_depth = 5, min_samples_split = 25,
              min_weight_fraction_leaf = 0.05, ccp_alpha = 1e-2)

knn = KNeighborsClassifier(n_neighbors=10)

QDA = QuadraticDiscriminantAnalysis(reg_param = 0.8)

#glm.get_params()

heart = pd.read_csv('//Users//behroozkeshavarzi//FinalHeart.csv')
heart.drop(['Unnamed: 0'],axis=1,inplace=True)

X = heart.drop(['num'],axis=1).copy()
Y = heart.num.copy()

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size= 0.33, stratify=Y)

estimators = [('glm', make_pipeline(StandardScaler(),glm)),
              ('rfc', rfc),
              ('svc', make_pipeline(StandardScaler(),svc)),
              ('gbc', gbc),
              ('knn',knn)
             ]

clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), stack_method = 'predict_proba')
print('Stacking method for the case which has GLM, RFC, SVC, GBC, KNN and Logistic regression as the meta model')
model_fit2(clf, X_train, X_test, Y_train, Y_test)
print('*' * 200)

estimators = [('rfc', rfc),
              ('svc', make_pipeline(StandardScaler(),svc)),
              ('gbc', gbc),
              ('knn',knn),
              ('glm', make_pipeline(StandardScaler(),glm))
             ]


clf = StackingClassifier(estimators=estimators, final_estimator=RandomForestClassifier(), stack_method = 'predict_proba')
print('Stacking method for the case which has GLM, RFC, SVC, GBC, KNN and Random forest as the meta model')
model_fit2(clf, X_train, X_test, Y_train, Y_test)
print('*' * 200)

estimators = [('glm', make_pipeline(StandardScaler(),glm)),
              ('rfc', rfc),
              ('svc', make_pipeline(StandardScaler(),svc)),
              ('gbc', gbc),
              ('knn',knn),
              ('QDA',make_pipeline(StandardScaler(),QDA))
             ]

clf = StackingClassifier(estimators=estimators, final_estimator=RandomForestClassifier())
print('Stacking method for the case which has GLM, RFC, SVC, GBC, KNN, and QDA and Random forest as the meta model')
model_fit2(clf, X_train, X_test, Y_train, Y_test)
print('*' * 200)

clf = StackingClassifier(estimators=estimators, final_estimator=svm.SVC())
print('Stacking method for the case which has GLM, RFC, SVC, GBC, KNN, and QDA and SVC as the meta model')
model_fit2(clf, X_train, X_test, Y_train, Y_test)
