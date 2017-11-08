import pandas as pd
import numpy as np
import gc

from data_exploration import visualize_distribution

def load_full_data():

    # Load the Datasets #
    # We need to load the datasets that will be needed to train our machine learning algorithms,
    #  handle our data and make predictions.
    # Note that these datasets are the ones that are already provided once you enter the
    # competition by accepting terms and conditions

    train_2016 = pd.read_csv('./data/train_2016_v2.csv', parse_dates=["transactiondate"])
    properties_2016 = pd.read_csv('./data/properties_2016.csv')

    train_2017 = pd.read_csv('./data/train_2017.csv', parse_dates=["transactiondate"])
    properties_2017 = pd.read_csv('./data/properties_2017.csv')

    train_temp = [train_2016 ,train_2017]
    properties_temp = [properties_2016, properties_2017]
    train = pd.concat(train_temp)
    properties = pd.concat(properties_temp)


    test = pd.read_csv('./data/sample_submission.csv')
    test = test.rename(
        columns={'ParcelId': 'parcelid'})  # To make it easier for merging datasets on same column_id later

    # Analyse the Dimensions of our Datasets.
    print("Training Size:" + str(train.shape))
    print("Property Size:" + str(properties.shape))
    print("Sample Size:" + str(test.shape))


    # Type Converting the DataSet
    # The processing of some of the algorithms can be made quick if data
    # representation is made in int/float32 instead of int/float64.
    # Therefore, in order to make sure that all of our columns types are in
    # float32, we are implementing the following lines of code #


    for c, dtype in zip(properties.columns, properties.dtypes):
        if dtype == np.float64:
            properties[c] = properties[c].astype(np.float32)
        if dtype == np.int64:
            properties[c] = properties[c].astype(np.int32)

    for column in test.columns:
        if test[column].dtype == int:
            test[column] = test[column].astype(np.int32)
        if test[column].dtype == float:
            test[column] = test[column].astype(np.float32)


    ###Merging the Datasets ###
    # We are merging the properties dataset with training and testing dataset for model building and testing prediction #

    df_train = train.merge(properties, how='left', on='parcelid')
    df_test = test.merge(properties, how='left', on='parcelid')

    visualize_distribution(properties,df_train,'calculatedfinishedsquarefeet')
    visualize_distribution(properties, df_train, 'structuretaxvaluedollarcnt')

    if 0:
        # creating a sub sample of data
        df_train_small = df_train[:10000]
        df_test_small = df_test[:5000]

        df_train = df_train_small
        df_test = df_test_small

        df_train.to_pickle('./cache/small_train.pkl')
        df_test.to_pickle('./cache/small_test.pkl')

    ### Remove previous variables to keep some memory
    del properties, train
    gc.collect()

    return df_train, df_test


def load_2016_data():

    # Load the Datasets #
    # We need to load the datasets that will be needed to train our machine learning algorithms,
    #  handle our data and make predictions.
    # Note that these datasets are the ones that are already provided once you enter the
    # competition by accepting terms and conditions

    train_2016 = pd.read_csv('./data/train_2016_v2.csv', parse_dates=["transactiondate"])
    properties_2016 = pd.read_csv('./data/properties_2016.csv')

    train = train_2016
    properties = properties_2016


    test = pd.read_csv('./data/sample_submission.csv')
    test = test.rename(
        columns={'ParcelId': 'parcelid'})  # To make it easier for merging datasets on same column_id later

    # Analyse the Dimensions of our Datasets.
    print("Training Size:" + str(train.shape))
    print("Property Size:" + str(properties.shape))
    print("Sample Size:" + str(test.shape))


    # Type Converting the DataSet
    # The processing of some of the algorithms can be made quick if data
    # representation is made in int/float32 instead of int/float64.
    # Therefore, in order to make sure that all of our columns types are in
    # float32, we are implementing the following lines of code #


    for c, dtype in zip(properties.columns, properties.dtypes):
        if dtype == np.float64:
            properties[c] = properties[c].astype(np.float32)
        if dtype == np.int64:
            properties[c] = properties[c].astype(np.int32)

    for column in test.columns:
        if test[column].dtype == int:
            test[column] = test[column].astype(np.int32)
        if test[column].dtype == float:
            test[column] = test[column].astype(np.float32)


    ###Merging the Datasets ###
    # We are merging the properties dataset with training and testing dataset for model building and testing prediction #

    df_train = train.merge(properties, how='left', on='parcelid')
    df_test = test.merge(properties, how='left', on='parcelid')

    visualize_distribution(properties,df_train,'calculatedfinishedsquarefeet')
    visualize_distribution(properties, df_train, 'structuretaxvaluedollarcnt')

    if 0:
        # creating a sub sample of data
        df_train_small = df_train[:10000]
        df_test_small = df_test[:5000]

        df_train = df_train_small
        df_test = df_test_small

        df_train.to_pickle('./cache/small_train.pkl')
        df_test.to_pickle('./cache/small_test.pkl')

    ### Remove previous variables to keep some memory
    del properties, train
    gc.collect()

    return df_train, df_test

def load_small_data():
    df_train = pd.read_pickle('./cache/small_train.pkl')
    df_test = pd.read_pickle('./cache/small_test.pkl')

    return df_train,df_test
