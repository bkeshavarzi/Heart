df = heartData.dropna()

fig, axis = plt.subplots(3,2,figsize=(27,16)) #age, cp, trestbps, restecg, thalach, exang

sns.histplot(data = heartData, x = 'age', hue = 'sex',ax = axis[0,0])
axis[0,0].set_xlabel('Samples Age',fontsize=16)
axis[0,0].set_ylabel('Frequency',fontsize=16)

sns.histplot(data = heartData, x = 'cp', hue = 'sex',ax = axis[0,1])
axis[0,1].set_xlabel('Chest Pain Type',fontsize=16)
axis[0,1].set_ylabel('Frequency',fontsize=16)
axis[0,1].set_xticklabels(['Typical Angina','Atypical Angina','Non-anginal Pain','Asymptomatic'])

sns.histplot(data = heartData, x = 'trestbps', hue = 'sex', ax = axis[1,0])
axis[1,0].set_xlabel('Resting Blood Pressure',fontsize=16)
axis[1,0].set_ylabel('Frequency',fontsize=16)

sns.histplot(data = heartData, x = 'restecg', hue = 'sex', ax = axis[1,1])
axis[1,1].set_xlabel('Resting Electrocardiographic Results',fontsize=16)
axis[1,1].set_ylabel('Frequency',fontsize=16)

sns.histplot(data = heartData, x = 'thalach', hue = 'sex', ax = axis[2,0])
axis[2,0].set_xlabel('Maximum Heart Rate Achieved',fontsize=16)
axis[2,0].set_ylabel('Frequency',fontsize=16)

sns.histplot(data = heartData, x = 'exang', hue = 'sex', ax = axis[2,1])
axis[2,1].set_xlabel('Exercise Induced Angina',fontsize=16)
axis[2,1].set_ylabel('Frequency',fontsize=16);


fig, axis = plt.subplots(1,3,figsize=(24,7))

sns.barplot(data = imp_df, x = 'index', y = 'knn_1', color = 'red', ax = axis[0], label = 'KNN (Neighbors = 1)')
axis[0].set_xlabel('Feature', fontsize = 20)
axis[0].set_ylabel('MSE', fontsize = 20)
axis[0].set_xticklabels(axis[0].get_xticklabels(),fontsize=14, rotation = 90)
axis[0].tick_params(axis='both',labelsize=16)
axis[0].legend(fontsize = 16)

sns.barplot(data = imp_df, x = 'index', y = 'knn_2', color = 'blue', ax = axis[1], label = 'KNN (Neighbors = 2)')
axis[1].set_xlabel('Feature', fontsize = 20)
axis[1].set_ylabel('MSE', fontsize = 20)
axis[1].set_xticklabels(axis[1].get_xticklabels(),fontsize=14, rotation = 90)
axis[1].tick_params(axis='both',labelsize=16)
axis[1].legend(fontsize = 16)

sns.barplot(data = imp_df, x = 'index', y = 'knn_3', color = 'orange', ax = axis[2], label = 'KNN (Neighbors = 3)')
axis[2].set_xlabel('Feature', fontsize = 20)
axis[2].set_ylabel('MSE', fontsize = 20)
axis[2].set_xticklabels(axis[2].get_xticklabels(),fontsize=14, rotation = 90)
axis[2].tick_params(axis='both',labelsize=16)
axis[2].legend(fontsize = 16);

axis[0].set_ylim([0,80])
axis[1].set_ylim([0,80])
axis[2].set_ylim([0,80])

fig, axis = plt.subplots(1,3,figsize=(24,7))

sns.barplot(data = imp_df, x = 'index', y = 'knn_4', color = 'red', ax = axis[0], label = 'KNN (Neighbors = 4)')
axis[0].set_xlabel('Feature', fontsize = 20)
axis[0].set_ylabel('MSE', fontsize = 20)
axis[0].set_xticklabels(axis[0].get_xticklabels(),fontsize=14, rotation = 90)
axis[0].tick_params(axis='both',labelsize=16)
axis[0].legend(fontsize = 16)

sns.barplot(data = imp_df, x = 'index', y = 'knn_5', color = 'blue', ax = axis[1], label = 'KNN (Neighbors = 5)')
axis[1].set_xlabel('Feature', fontsize = 20)
axis[1].set_ylabel('MSE', fontsize = 20)
axis[1].set_xticklabels(axis[1].get_xticklabels(),fontsize=14, rotation = 90)
axis[1].tick_params(axis='both',labelsize=16)
axis[1].legend(fontsize = 16)

sns.barplot(data = imp_df, x = 'index', y = 'knn_6', color = 'orange', ax = axis[2], label = 'KNN (Neighbors = 6)')
axis[2].set_xlabel('Feature', fontsize = 20)
axis[2].set_ylabel('MSE', fontsize = 20)
axis[2].set_xticklabels(axis[2].get_xticklabels(),fontsize=14, rotation = 90)
axis[2].tick_params(axis='both',labelsize=16)
axis[2].legend(fontsize = 16);

axis[0].set_ylim([0,80])
axis[1].set_ylim([0,80])
axis[2].set_ylim([0,80])