import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sklearn
import scipy

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from PlottingFunction import lda_1Dplot, plotModel, data_1Dplot

from Classification import model_fit, model_fit2, plot_confusion_matrix

from sklearn.preprocessing import StandardScaler

heart = pd.read_csv('//Users//behroozkeshavarzi//FinalHeart.csv')
heart.drop(['Unnamed: 0'],axis=1,inplace=True)

X = heart.drop(['num'],axis=1).copy()
Y = heart.num.copy()
SC = StandardScaler()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                        test_size = 0.33, random_state = 144, shuffle = True, stratify = Y)
X_train_scaled, X_test_scaled, Y_train_scaled, Y_test_Scaled = train_test_split(SC.fit_transform(X), Y,
                                                        test_size = 0.33, random_state = 144, shuffle = True, stratify = Y)

glm = LogisticRegression(max_iter = 1e7)
model_fit2(glm, X_train, X_test, Y_train, Y_test)
model_fit2(glm, X_train_scaled, X_test_scaled, Y_train_scaled, Y_test_Scaled, "Scaled Data")

