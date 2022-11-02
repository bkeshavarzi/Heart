def cv_fun(model, X, Y, param_grid):

    from sklearn.model_selection import GridSearchCV
    import time
    import pandas as pd
    model_grid = GridSearchCV(model, param_grid, cv = 3, return_train_score = True)
    model_grid.fit(X,Y)
    return pd.DataFrame(model_grid.cv_results_)

def model_fit(model, X_train, X_test, Y_train, Y_test, st = ""):
    
    import numpy as np
    import sklearn
    from sklearn.model_selection import cross_val_score

    print(str(type(model)).split('.')[-1].split("\'")[0])
    print(st)

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
    import warnings
    warnings.filterwarnings("ignore")
    
    print(str(type(model)).split('.')[-1].split("\'")[0])
    st = str(type(model)).split('.')[-1].split("\'")[0]

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
    ax[0].set_ylabel('Measured', fontsize= 18)
    ax[0].set_xlabel('Predicted', fontsize= 18)
    ax[0].set_title('Train Set',fontsize= 22)

    ax[1].tick_params(axis='both',labelsize=16)
    ax[1].set_ylabel('Measured', fontsize= 18)
    ax[1].set_xlabel('Predicted', fontsize= 18)
    ax[1].set_title('Test Set',fontsize= 22);

    fig, ax = plt.subplots(1,2,figsize= (18,6))
    sns.heatmap(conf_train_norm, ax = ax[0], annot=True)
    sns.heatmap(conf_test_norm,  ax = ax[1], annot=True)

    ax[0].tick_params(axis='both',labelsize=16)
    ax[0].set_ylabel('Measured', fontsize= 18)
    ax[0].set_xlabel('Predicted', fontsize= 18)
    ax[0].set_title('Train Set (Percent)',fontsize= 22)

    ax[1].tick_params(axis='both',labelsize=16)
    ax[1].set_ylabel('Measured', fontsize= 18)
    ax[1].set_xlabel('Predicted', fontsize= 18)
    ax[1].set_title('Test Set (Percent)',fontsize= 22);

def model_fit2(model, X_train, X_test, Y_train, Y_test, st = ""):
    
    import numpy as np
    import sklearn
    from sklearn.model_selection import cross_val_score

    print(str(type(model)).split('.')[-1].split("\'")[0])
    print(st)
    
    model.fit(X_train, Y_train)
    r2_train = np.round(model.score(X_train, Y_train),3)
    r2_test  = np.round(model.score(X_test, Y_test),3)
    print('Train score is :' + str(r2_train))
    print('Test score is :' + str(r2_test))

def plot_prob(df, model, xfeature, yfeature, X, Y, iClass):

    import numpy as np
    import pandas as pd 
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(1,1,figsize=(12,8))
    sns.scatterplot(data = df, x = xfeature, y = yfeature, hue = 'num', palette='RdBu', s=120, ax = ax)
    ax.tick_params(axis='both',labelsize=16)
    ax.set_xlabel(xlabel = xfeature, fontsize=16)
    ax.set_ylabel(ylabel = yfeature, fontsize=16)

    n_distinct = len(np.unique(Y))
    col_name = list(df.columns) + ['Class'+str(j) for j in range(n_distinct)] 
    data = np.hstack([df.values, model.predict_proba(X)])
    df2 = pd.DataFrame(data, columns = col_name)

    ax2 = ax.twinx()
    sns.lineplot(data = df2, x = xfeature, y = 'Class'+str(iClass), color = 'red', label = iClass, ax = ax2)
    ax2.tick_params(axis='y',labelsize = 16)
    ax2.set_ylabel('Probability for Class = ' + str(iClass), fontsize = 16)
    plt.legend(loc='upper left');

def get_precision_recall(model, X_train, X_test, Y_train, Y_test, iClass):

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import sklearn
    from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, roc_curve, roc_auc_score
    from sklearn.model_selection import cross_val_predict
    import warnings
    warnings.filterwarnings("ignore")

    try:
        Y_scores_train = cross_val_predict(model, X_train, Y_train, cv = 3, method = "decision_function")
        Y_scores_test  = cross_val_predict(model, X_test, Y_test,   cv = 3, method = "decision_function")
    except:
        Y_scores_train = cross_val_predict(model, X_train, Y_train, cv = 3, method = "predict_proba")
        Y_scores_test  = cross_val_predict(model, X_test, Y_test,   cv = 3, method = "predict_proba")

    Y_scores_train = Y_scores_train[:,iClass]
    Y_scores_test  = Y_scores_test[:,iClass]
    Y_train_class = Y_train == iClass
    Y_test_class = Y_test == iClass
    
    precision_train = precision_score(Y_train_class, model.predict(X_train)==iClass)
    precision_test  = precision_score(Y_test_class, model.predict(X_test)==iClass)

    recall_train    = recall_score(Y_train_class, model.predict(X_train)==iClass)
    recall_test     = recall_score(Y_test_class, model.predict(X_test)==iClass)

    f1_train  = f1_score(Y_train_class, model.predict(X_train)==iClass)
    f1_test  = f1_score(Y_test_class, model.predict(X_test)==iClass)

    print('Precision score for training set is :' + str(np.round(precision_train,2)))
    print('Precision score for testing  set is :' + str(np.round(precision_test,2)))

    print('Recall score for training set is :' + str(np.round(recall_train,2)))
    print('Recall score for testing  set is :' + str(np.round(recall_test,2)))

    print('F1 score for training set is :' + str(np.round(f1_train,2)))
    print('F1 score for testing  set is :' + str(np.round(f1_test,2)))
    
    precision_train, recall_train, threshold_train = precision_recall_curve(Y_train_class, Y_scores_train)
    precision_test,  recall_test,  threshold_test  = precision_recall_curve(Y_test_class, Y_scores_test)

    data_train = np.hstack([precision_train[:-1].reshape(-1,1), recall_train[:-1].reshape(-1,1), threshold_train.reshape(-1,1)])
    data_test  = np.hstack([precision_test[:-1].reshape(-1,1),  recall_test[:-1].reshape(-1,1),  threshold_test.reshape(-1,1)])

    df_train = pd.DataFrame(data = data_train, columns=['Precision', 'Recall', 'Threshold'])
    df_test  = pd.DataFrame(data = data_test,  columns=['Precision', 'Recall', 'Threshold'])

    fig, axis = plt.subplots(1,2,figsize=(18,8))
    sns.lineplot(data = df_train, x = 'Threshold', y = 'Precision', color = 'red', marker='o', label = 'Precision', ax = axis[0])
    sns.lineplot(data = df_train, x = 'Threshold', y = 'Recall', color = 'blue', marker='s', label = 'Recall', ax = axis[0])
    axis[0].set_xlabel('Threshold', fontsize = 18)
    axis[0].set_ylabel('Score (Training Set)', fontsize = 18)
    axis[0].tick_params(axis='both',labelsize = 16)
    axis[0].set_title('Precision, Recall curve for Class ' + str(iClass), fontsize = 20)
    axis[0].grid()

    sns.lineplot(data = df_test, x = 'Threshold', y = 'Precision', color = 'red', marker='o', label = 'Precision', ax = axis[1])
    sns.lineplot(data = df_test, x = 'Threshold', y = 'Recall', color = 'blue', marker='s', label = 'Recall', ax = axis[1])
    axis[1].set_xlabel('Threshold', fontsize = 18)
    axis[1].set_ylabel('Score (Test set)', fontsize = 18)
    axis[1].tick_params(axis='both',labelsize = 16)
    axis[1].set_title('Precision, Recall curve for Class ' + str(iClass), fontsize = 20)
    axis[1].grid()

    fig, axis = plt.subplots(1,2,figsize=(18,8))
    sns.lineplot(data = df_train, x = 'Recall', y = 'Precision', color = 'red', marker='o', label = 'Train', ax = axis[0])
    sns.lineplot(data = df_test, x = 'Recall', y = 'Precision', color = 'blue', marker='s', label = 'Test', ax = axis[0])
    axis[0].set_xlabel('Recall', fontsize = 18)
    axis[0].set_ylabel('Precision', fontsize = 18)
    axis[0].tick_params(axis='both',labelsize = 16)
    axis[0].set_title('Recall vs. Precision curve for Class ' + str(iClass), fontsize = 20)
    axis[0].grid()

    fpr_train, tpr_train, threshold_train = roc_curve(Y_train_class, Y_scores_train)
    fpr_test,  tpr_test,  threshold_test =  roc_curve(Y_test_class, Y_scores_test)

    sns.lineplot( x = fpr_train, y = tpr_train, color = 'red', marker='o', label = 'Train', ax = axis[1])
    sns.lineplot( x = fpr_test,  y = tpr_test,  color = 'blue', marker='s', label = 'Test', ax = axis[1])
    axis[1].set_xlabel('Recall', fontsize = 18)
    axis[1].set_ylabel('Precision', fontsize = 18)
    axis[1].tick_params(axis='both',labelsize = 16)
    axis[1].set_title('Receiver Operating Characteristic Curve for Class ' + str(iClass), fontsize = 20)
    axis[1].grid()

    train_auc = roc_auc_score(Y_train_class, Y_scores_train)
    test_auc  = roc_auc_score(Y_test_class, Y_scores_test)

    print('AUC for training set is :' + str(np.round(train_auc,2)))
    print('AUC for testing  set is :' + str(np.round(test_auc,2)))









