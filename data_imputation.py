import numpy as np
import pandas as pd
import matplotlib as plt


def data_imputation(df_train, df_test):
    # imputing missing values with corresponding data to be filled

    mode_imputations = ['airconditioningtypeid', 'heatingorsystemtypeid', 'fireplacecnt']
    for element in mode_imputations:
        mode_element = df_test[element].mode()
        df_test[element] = df_test[element].fillna(mode_element)

    zero_imputations = ['poolcnt', 'garagecarcnt']
    for element in zero_imputations:
        df_test[element] = df_test[element].fillna(0)

    mean_imputations = ['fullbathcnt']
    for element in mean_imputations:
        mean_element = df_test[element].mean()
        df_test[element] = df_test[element].fillna(mean_element)

    # Room imputations



