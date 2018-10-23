'''
modeling 

cross validation
'''

import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV

def model():
	



def k_fold():
	xg_gsearch = GridSearchCV(estimator=xgb.XGBClassifier(),
								param_grid = param,
								scoring = 'roc_auc',
								n_jobs=2,
								lid=False,
								cv=5,
								verbose=2
								)
	xg_g

