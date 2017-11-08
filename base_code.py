############################################
# NCSU ALDA FALL 2017 - Project Work
# Zillow House Price Estimation
# Authors - Harish Pullagurla ( hpullag@ncsu.edu), Pooja Mehta , Ratika Kapoor
# Created - 10/30/2017
############################################


# Importing Libraries or Packages that are needed throughout the Program
import numpy as np
import pandas as pd
import xgboost as  xgb

import matplotlib.pyplot as plt
import seaborn as sns

from data_exploration import *
from data_loader import *
from feature_engineering import *
from ml_models import *

sns.set(color_codes=True)

DEBUG = 1



def main():

    # df_train, df_test = load_full_data()
    df_train, df_test = load_2016_data()
    # df_train, df_test = load_small_data()
    data_exploration(df_train)

    df_train,df_test = data_preprocessing(df_train,df_test)

    x_train, y_train, x_test,k = feature_selection(df_train,df_test)
    model_experiments(x_train,y_train,x_test)



if __name__ == "__main__":
    main()
