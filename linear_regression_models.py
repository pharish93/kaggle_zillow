import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

def linear_reg_model(x_train,y_train,x_test):
    ### Cross Validation ###

    # We are dividing our datasets into the training and validation sets so that
    # we could monitor and the test the progress of our machine learning algorithm.
    # This would let us know when our model might be over or under fitting on the
    # dataset that we have employed. #

    X = x_train.values
    y = y_train.values

    Xtrain, Xvalid, ytrain, yvalid = train_test_split(X, y, test_size=0.2, random_state=42)



    Predicted_test_xgb = 0

    return Predicted_test_xgb


