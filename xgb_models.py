import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.metrics import r2_score

def xgb_model_experiments(x_train,y_train,x_test):
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

    # We can now select the parameters for Xgboost and monitor the progress of results
    # on our validation set. The explanation of the xgboost parameters and what they do
    # can be found on the following link http://xgboost.readthedocs.io/en/latest/parameter.html #

    dtrain = xgb.DMatrix(Xtrain, label=ytrain)
    dvalid = xgb.DMatrix(Xvalid, label=yvalid)
    dtest = xgb.DMatrix(x_test.values)

    # Try different parameters!
    xgb_params = {'min_child_weight': 5, 'eta': 0.035, 'colsample_bytree': 0.5, 'max_depth': 4,
                'subsample': 0.85, 'lambda': 0.8, 'nthread': -1, 'booster' : 'gbtree', 'silent': 1, 'gamma' : 0,
                'eval_metric': 'mae', 'objective': 'reg:linear' }

    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

    model_xgb = xgb.train(xgb_params, dtrain, 1000, watchlist, early_stopping_rounds=100,
                      maximize=False, verbose_eval=10)

    # Predicting the results
    # Let us now predict the target variable for our test dataset.
    # All we have to do now is just fit the already trained model on the test set
    # that we had made merging the sample file with properties dataset #

    Predicted_test_xgb = model_xgb.predict(dtest)
    # print Predicted_test_xgb
    # np.savetxt('./cache/y_pred.txt',Predicted_test_xgb)

    return Predicted_test_xgb

    # Submitting the Results
    # Once again load the file and start submitting the results in each column

    # sample_file = pd.read_csv('./data/sample_submission.csv')
    # for c in sample_file.columns[sample_file.columns != 'ParcelId']:
    #     sample_file[c] = Predicted_test_xgb

    # print('Preparing the csv file ...')
    # sample_file.to_csv('xgb_predicted_results.csv', index=False, float_format='%.4f')
    # print("Finished writing the file")

    # return predicted values

