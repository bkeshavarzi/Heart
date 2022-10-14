def modelFit(model, X_train, Y_train, X_test, Y_test):

    print(type(model))
    model.fit(X_train, Y_train)
    import numpy as np
    r2_train = np.round(model.score(X_train, Y_train),3)
    r2_test  = np.round(model.score(X_test, Y_test),3)
    print('Train score is :' + str(r2_train))
    print('Test score is :' + str(r2_test))

def plot_confusion(model, X_train, X_test, Y_train, Y_test):

    from sklearn.model_selection import confusion_matrix
    import matplotlib.pyplot as plt
    Y_train_pred = model.predict(X_train)
    Y_test_pred = model.predict(X_test)

    conf_data_train = confusion_matrix(Y_train,Y_train_pred)
    conf_data_test  = confusion_matrix(Y_test,Y_test_pred)
    conf_train_norm = np.zeros_like(conf_data_train, dtype=float)
    conf_test_norm = np.zeros_like(conf_data_test, dtype=float)
    sum_vec_train = np.sum(conf_data_train, axis = 1)
    sum_vec_test  = np.sum(conf_data_test , axis = 1)

    for irow in range(conf_train_norm.shape[0]):

        conf_train_norm[irow,:] = 100 * np.round(conf_data_train[irow,:]/sum_vec_train[irow],2)
        conf_test_norm[irow,:]  = 100 * np.round(conf_data_test[irow,:]/sum_vec_test[irow],2)

    fig, ax = plt.subplots(1,2,figsize= (18,6))
    sns.heatmap(conf_train_norm, ax = ax[0], annot=True)
    sns.heatmap(conf_test_norm,  ax = ax[1], annot=True)

    ax[0].tick_params(axis='both',labelsize=16)
    ax[0].set_xlabel('Measured', fontsize= 18)
    ax[0].set_ylabel('Predicted', fontsize= 18)
    ax[0].set_title('Train Set',fontsize= 22)

    ax[1].tick_params(axis='both',labelsize=16)
    ax[1].set_xlabel('Measured', fontsize= 18)
    ax[1].set_ylabel('Predicted', fontsize= 18)
    ax[1].set_title('Test Set',fontsize= 22);