import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt


def linear_reg_model(x_train,y_train,x_test):
    ### Cross Validation ###

    # We are dividing our datasets into the training and validation sets so that
    # we could monitor and the test the progress of our machine learning algorithm.
    # This would let us know when our model might be over or under fitting on the
    # dataset that we have employed. #

    print "Linear Regression Modeling Function entered"

    X = x_train.values
    y = y_train.values

    lr = linear_model.LinearRegression()

    lr.fit(X, y)

    Predicted_test = lr.predict(x_test.values)

    print "Linear Regression Modeling Function Completed"

    return Predicted_test


