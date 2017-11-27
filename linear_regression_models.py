import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
from sklearn import decomposition
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

def pca_data(x_train,x_test):

    pca = decomposition.PCA(n_components=3)

    pca.fit(x_train)
    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)

    return x_train,x_test

def linear_reg_model(x_train,y_train,x_test):

    print "Linear Regression Modeling Function entered"

    x_train_vals = x_train.values
    y = y_train.values

    X,x_test_vals = pca_data(x_train_vals,x_test.values)

    lr = linear_model.Ridge (alpha = .5)
    lr.fit(X, y)

    Predicted_test = lr.predict(x_test_vals)

    print "Linear Regression Modeling Function Completed"

    return Predicted_test

def random_forest_reg(x_train,y_train,x_test):

    print "Random Forest Regressor "

    X = x_train.values
    y = y_train.values

    regr = RandomForestRegressor(n_estimators=250)
    regr.fit(X, y)
    Predicted_test = regr.predict(x_test.values)
    return Predicted_test

