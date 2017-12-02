import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

def model_experiments(x_train,y_train,x_test):
    ### Cross Validation ###

    # We are dividing our datasets into the training and validation sets so that
    # we could monitor and the test the progress of our machine learning algorithm.
    # This would let us know when our model might be over or under fitting on the
    # dataset that we have employed. #

    from sklearn.model_selection import train_test_split

    X = x_train.values
    y = y_train.values

    Xtrain, Xvalid, ytrain, yvalid = train_test_split(X, y, test_size=0.2, random_state=42)

    #Implement the Xgboost#

    # We select the parameters for Xgboost and monitor the progress of results
    # on our validation set.
    dtrain = xgb.DMatrix(Xtrain, label=ytrain)
    dvalid = xgb.DMatrix(Xvalid, label=yvalid)
    dtest = xgb.DMatrix(x_test.values)

    # Try different parameters!
    xgb_params = {'min_child_weight': 5, 'eta': 0.035, 'colsample_bytree': 0.5, 'max_depth': 8,
                'subsample': 0.85, 'lambda': 0.8, 'nthread': -1, 'booster' : 'gbtree', 'silent': 1, 'gamma' : 0,
                'eval_metric': 'mae', 'objective': 'reg:linear' }

    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

    model_xgb = xgb.train(xgb_params, dtrain, 10000, watchlist, early_stopping_rounds=100,
                      maximize=False, verbose_eval=10)

    # Predicting the results

    Predicted_test_xgb = model_xgb.predict(dtest)
    # print Predicted_test_xgb
    # np.savetxt('./cache/y_pred.txt',Predicted_test_xgb)

    return Predicted_test_xgb


