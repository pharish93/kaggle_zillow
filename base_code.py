############################################
# NCSU ALDA FALL 2017 - Project Work
# Zillow House Price Estimation
# Authors - Harish Pullagurla ( hpullag@ncsu.edu), Pooja Mehta , Ratika Kapoor
# Created - 10/30/2017
############################################


# Importing Libraries or Packages that are needed throughout the Program
import numpy as np
import pandas as pd

from data_exploration import *
from data_loader import *
from feature_engineering import *
from xgb_models import *
from linear_regression_models import *
from testing_metric import *
from svr_model import *

sns.set(color_codes=True)

DEBUG = 0

def main():

    # Step1 :- Load the data and visualize distributions
    # df_train, df_test = load_full_data()
    # df_train, df_test = load_2016_data()
    # df_train, df_test = load_small_data()
    df_train, df_test = load_train_split_test()

    # Step 2 : Data Exploration
    if DEBUG:
        data_exploration(df_train)

    # Step 3 : Feature Understanding and modification
    x_train,x_test,y_train,y_test = feature_engineering(df_train,df_test)

    # Step 4  : Sending data to Machine Learning Model
    # y_pred = xgb_model_experiments(x_train,y_train,x_test)

    # y_pred = random_forest_reg(x_train, y_train, x_test)

    y_pred = linear_reg_model(x_train, y_train, x_test)

    # y_pred = svm_model_experiments(x_train, y_train, x_test)


    # Step 5 : Testing Error Metrices
    error_metric_calc(y_test.values, y_pred)

if __name__ == "__main__":
    main()
