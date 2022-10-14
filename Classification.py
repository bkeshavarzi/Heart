def cv_fun(model, X, Y, param_grid):

    from sklearn.model_selection import GridSearchCV
    import time
    import pandas as pd
    model_grid = GridSearchCV(model, param_grid, cv = 3, return_train_score = True)
    model_grid.fit(X,Y)
    return pd.DataFrame(model_grid.cv_results_)

def model_fit(model, X_train, X_test, Y_train, Y_test):
    
    import numpy as np
    import sklearn
    from sklearn.model_selection import cross_val_score

    print(str(type(model)).split('.')[-1].split("\'")[0])
    model.fit(X_train, Y_train)
    r2_train = np.round(model.score(X_train, Y_train),3)
    r2_test  = np.round(model.score(X_test, Y_test),3)
    print('Train score is :' + str(r2_train))
    print('Test score is :' + str(r2_test))
    print(cross_val_score(model, X_train, Y_train, cv = 3))
    print(cross_val_score(model, X_test, Y_test, cv = 3))

def plot_confusion_matrix(model, X_train, X_test, Y_train, Y_test):

    import sklearn
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt 
    import numpy as np

    print(str(type(model)).split('.')[-1].split("\'")[0])

    Y_train_pred = model.predict(X_train)
    Y_test_pred  = model.predict(X_test)

    conf_data_train = confusion_matrix(Y_train, Y_train_pred)
    conf_data_test  = confusion_matrix(Y_test, Y_test_pred)
    conf_train_norm = np.zeros_like(conf_data_train, dtype=float)
    conf_test_norm = np.zeros_like(conf_data_test, dtype=float)

    sum_vec_train = np.sum(conf_data_train, axis = 1)
    sum_vec_test  = np.sum(conf_data_test , axis = 1)

    for irow in range(conf_train_norm.shape[0]):

        conf_train_norm[irow,:] = 100 * np.round(conf_data_train[irow,:]/sum_vec_train[irow],2)
        conf_test_norm[irow,:]  = 100 * np.round(conf_data_test[irow,:]/sum_vec_test[irow],2)

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

    fig, ax = plt.subplots(1,2,figsize= (18,6))
    sns.heatmap(conf_train_norm, ax = ax[0], annot=True)
    sns.heatmap(conf_test_norm,  ax = ax[1], annot=True)

    ax[0].tick_params(axis='both',labelsize=16)
    ax[0].set_xlabel('Measured', fontsize= 18)
    ax[0].set_ylabel('Predicted', fontsize= 18)
    ax[0].set_title('Train Set (Percent)',fontsize= 22)

    ax[1].tick_params(axis='both',labelsize=16)
    ax[1].set_xlabel('Measured', fontsize= 18)
    ax[1].set_ylabel('Predicted', fontsize= 18)
    ax[1].set_title('Test Set (Percent)',fontsize= 22);

def model_fit2(model, X_train, X_test, Y_train, Y_test):
    
    import numpy as np
    import sklearn
    from sklearn.model_selection import cross_val_score

    print(str(type(model)).split('.')[-1].split("\'")[0])
    model.fit(X_train, Y_train)
    r2_train = np.round(model.score(X_train, Y_train),3)
    r2_test  = np.round(model.score(X_test, Y_test),3)
    print('Train score is :' + str(r2_train))
    print('Test score is :' + str(r2_test))