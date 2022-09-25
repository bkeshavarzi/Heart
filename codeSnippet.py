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