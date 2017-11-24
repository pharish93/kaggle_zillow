import numpy as np
import pandas as pd
import matplotlib as plt


def data_imputation(df_train, df_test):
    # imputing missing values with corresponding data to be filled

    mode_imputations = ['airconditioningtypeid', 'heatingorsystemtypeid', 'roomcnt','fireplacecnt']
    for element in mode_imputations:
        mode_element = df_test[element].mode()
        df_test[element] = df_test[element].fillna(mode_element)


    zero_imputations = ['poolcnt','garagecarcnt']
    for element in zero_imputations:
        df_test[element] = df_test[element].fillna(0)

    missing_df = df_train.isnull().sum(axis=0).reset_index()
    missing_df.columns = ['column_name', 'missing_count']
    missing_df['missing_ratio'] = missing_df['missing_count'] / df_train.shape[0]
