def cv_fun(model, X, Y, param_grid):

	from sklearn.model_selection import GridSearchCV
	import time
	import pandas as pd
	model_grid = GridSearchCV(model, param_grid, cv = 3, return_train_score = True)
	%time model_grid.fit(X,Y)
	return pd.DataFrame(model_grid.cv_results_)