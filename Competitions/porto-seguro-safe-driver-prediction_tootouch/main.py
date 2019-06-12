'''
import data
import train 

predict
'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import metrics


def onehotencoder(df, variables):
	dummies = pd.DataFrame()
	for var in variables:
		dummy = pd.get_dummies(df[var],prefix=var)
		dummies = pd.concat([dummies,dummy],axis=1)
		df = df.drop(var, axis=1)
	df = pd.concat([df,dummies],axis=1)
	
	return df

def gini(pred, actual, cmpcol=0, sortcol=1):
	if type(actual).__module__ == 'xgboost.core':
		actual = actual.get_label()
	if type(pred).__module__ == 'xgboost.core':
		pred = pred.get_label()
	
	all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
	all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
	totalLosses = all[:, 0].sum()
	giniSum = all[:, 0].cumsum().sum() / totalLosses

	giniSum -= (len(actual) + 1) / 2.
	return giniSum / len(actual)


def gini_normalized(p, a):
	return 'gini_normal', gini(p, a) / gini(a, a)


def modelfit(alg, dtrain, dval, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
	predictors = [x for x in dtrain.columns if x not in ['id','target']]
	if useTrainCV:
		xgb_param = alg.get_xgb_params()
		xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain['target'].values)
		cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
			metrics='auc', early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
		alg.set_params(n_estimators=cvresult.shape[0])
	
	#Fit the algorithm on the data
	alg.fit(dtrain[predictors], dtrain['target'],eval_metric='auc')
		
	#Predict training set:
	dtrain_predictions = alg.predict(dtrain[predictors])
	dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
	  
	#Predict training set:
	dval_predictions = alg.predict(dval[predictors])
	dval_predprob = alg.predict_proba(dval[predictors])[:,1]
	
	#Print model report:
	print("\nModel Report[train data]")
	print("Accuracy : %.4g" % metrics.accuracy_score(dtrain['target'].values, dtrain_predictions))
	print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['target'], dtrain_predprob))
	
	print("\nModel Report[validation data]")
	print("Accuracy : %.4g" % metrics.accuracy_score(dval['target'].values, dval_predictions))
	print("AUC Score (Train): %f" % metrics.roc_auc_score(dval['target'], dval_predprob))
	
	gini_pred = gini(dval_predprob, dval['target'])
	gini_max = gini(dval['target'], dval['target'])
	_, ngini = gini_normalized(dval_predprob, dval['target'])

	print('Gini: {0:.5f}, Max Gini: {1:.5f}, Normlization Gini: {2:.5f}'.format(gini_pred, gini_max, ngini))
	
	feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=True)
	f, ax = plt.subplots(1,1, figsize=(10,50))
	feat_imp.plot(kind='barh', title='Feature importance', ax=ax)
	ax.set_ylabel('Feature importance score')
	plt.show()
	
	return alg
	
def param_tune(df, param_test, xgb_param):
	predictors = [x for x in df.columns if x not in ['id','target']]
	param_test = param_test
	gsearch = GridSearchCV(estimator = xgb.XGBClassifier(**xgb_param, seed=1223), 
							param_grid = param_test, 
							scoring = 'roc_auc',
							n_jobs=2,
							iid=False,
							verbose=2,
							cv=5)
	gsearch.fit(df[predictors],df['target'])
	return gsearch

if __name__ == '__main__':

	train_origin = pd.read_csv('d:/Project/TT/kaggleml-master/code1/kaggle_porto-seguro-safe-driver-prediction/input/train.csv', na_values=['-1','-1.0'])
	test_set = pd.read_csv('d:/Project/TT/kaggleml-master/code1/kaggle_porto-seguro-safe-driver-prediction/input/test.csv', na_values=['-1','-1.0'])

	train_set = train_origin.copy()

	drop_feature = ['ps_ind_01', 'ps_car_11', 'ps_car_14', 'ps_car_15', 'ps_calc_01', 'ps_calc_03', 
				   'ps_calc_04', 'ps_calc_05', 'ps_calc_06', 'ps_calc_06', 'ps_calc_07', 'ps_calc_08',
				   'ps_calc_09', 'ps_calc_10', 'ps_calc_11', 'ps_calc_12', 'ps_calc_13', 'ps_calc_14',
				   'ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_13_bin', 'ps_calc_15_bin', 'ps_calc_16_bin',
				   'ps_calc_17_bin', 'ps_calc_18_bin', 'ps_calc_19_bin', 'ps_calc_20_bin']
	train_set = train_set.drop(drop_feature, axis=1)
	test_set = test_set.drop(drop_feature, axis=1)




	cat = [col for col in train_set.columns if 'cat' in col]
	train_origin = onehotencoder(train_origin, cat)
	train_t = onehotencoder(train_set, cat)
	test = onehotencoder(test_set,cat)

	x_data = train_t.drop(columns=['id','target'])
	y_data = train_t[['target']]

	test_idx = test['id']
	test = test.drop(columns=['id'])

	x_train, x_val, y_train, y_val = train_test_split(x_data,y_data,test_size=0.3,random_state=223)

	train = pd.concat([x_train,y_train], axis=1)
	val = pd.concat([x_val,y_val], axis=1)



	xgb4 = xgb.XGBClassifier(
		learning_rate =0.1,
		n_estimators=2000,
		max_depth=4,
		min_child_weight=1,
		gamma=0,
		subsample= 0.7,
		colsample_bytree=0.7,
		reg_alpha=22,
		nthread=4,
		scale_pos_weight=1,
		objective= 'gpu:binary:logistic',
		tree_method='gpu_hist',
		predictor='gpu_predictor',
		seed=1223)
	alg4 = modelfit(xgb4, train, val)



	filename = 'f_score3.csv'
	value = pd.Series(alg4.get_booster().get_fscore()).sort_values(ascending=False)
	ranking = pd.DataFrame({'id':value.index, 'value':value, 'rank':list(range(1,len(value)+1))})
	idx_df = pd.DataFrame({'id':train_origin.columns[2:], 'idx':list(range(1,len(train_origin.columns[2:])+1))})
	f_score_df = pd.merge(ranking,idx_df, how='right')
	f_score_df = f_score_df.sort_values(by='idx')
	f_score_df.to_csv('d:/Project/TT/kaggleml-master/code1/kaggle_porto-seguro-safe-driver-prediction/input/' + filename)


	predictors = [x for x in train_t.columns if x not in ['id','target']]
	alg4.fit(train_t[predictors], train_t['target'], eval_metric='auc')

	print('predict test set')
	test_pred = alg4.predict_proba(test)[:,1]
	print('save submission file')
	filename = 'submission8.csv'
	test_df = pd.DataFrame({'id':test_idx, 'target':test_pred})
	test_df.to_csv('d:/Project/TT/kaggleml-master/code1/kaggle_porto-seguro-safe-driver-prediction/input/' + filename,index=False)
	print('complete')



