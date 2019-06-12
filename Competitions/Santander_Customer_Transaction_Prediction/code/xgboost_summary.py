import pandas as pd
import numpy as np

import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.decomposition import PCA
from imblearn import over_sampling

from data import *

# Load data
data, test = loaddata()
x_data, y_data = data.iloc[:,1:], data['target']

# params
config = {
    "seed":2019,
    "k_folds":3,
    "early_stopping_rounds":100
}

params = {
    "learning_rate": 0.1,
    "n_estimators": 10000,
    "max_depth": 3,
    "min_child_weight": 5,
    "subsample": 1.0,
    "colsample_bytree": 0.5,
    "colsample_bylevel": 0.5,
    "alpha": 0,
    "lambda": 10,
    "objective": "gpu:binary:logistic",
    "tree_method": "gpu_hist",
    "predictor": "gpu_predictor",
    "eval_metric":"auc"
}

def predict(model, valid, test, best_iter, fold):
    valid_prob = model.predict(valid[0], ntree_limit=best_iter)
    auc = metrics.roc_auc_score(y_true=valid[1], y_score=valid_prob)
    print('valid auc: ',auc)

    test_prob = model.predict(test, ntree_limit=best_iter) / fold

    return auc, test_prob

# fitting
def fitting(x_data, y_data, params, config):
    folds = StratifiedKFold(n_splits=config['k_folds'], random_state=config['seed'], shuffle=True)

    auc_list = list()
    proba = np.zeros(len(test))
    xgtest = xgb.DMatrix(test)

    for i, (train_idx, valid_idx) in enumerate(folds.split(X=x_data, y=y_data)):
        print('='*25,' {} '.format(i) ,'='*25)
        x_train, y_train = x_data.iloc[train_idx, :], y_data[train_idx]
        x_valid, y_valid = x_data.iloc[valid_idx, :], y_data[valid_idx]

        # SMOTE
        smote = over_sampling.SMOTE(n_jobs=8, ratio=0.2)
        x_res, y_res = smote.fit_resample(x_train, y_train)
        x_train = pd.DataFrame(x_res, columns=x_data.columns)
        y_train = pd.Series(y_res)

        xgtrain = xgb.DMatrix(x_train, label=y_train)
        xgvalid = xgb.DMatrix(x_valid, label=y_valid)

        watchlist = [(xgtrain,'train'),(xgvalid, 'evel')]
        bst = xgb.train(params, xgtrain,
                        num_boost_round=params['n_estimators'],
                        evals=watchlist,
                        early_stopping_rounds=config['early_stopping_rounds'],
                        verbose_eval=100)

        auc, p = predict(bst, (xgvalid, y_valid), xgtest, bst.best_iteration, folds.n_splits)
        auc_list.append(auc)
        proba += p

    print('='*100)
    print('AUC_LIST')
    print(auc_list)

    print('-'*100)
    print('Mean AUC: {}'.format(np.mean(auc_list)))

    return proba

proba = fitting(x_data, y_data, params, config)

root_dir = os.path.abspath(os.path.join(os.getcwd(),'..'))
submit_dir = root_dir + '/submission'
id = len(os.listdir(submit_dir)) + 1
pd.DataFrame({'ID_code':test.index, 'target':proba}).to_csv('{}/s{}.csv'.format(submit_dir, id),index=False)
