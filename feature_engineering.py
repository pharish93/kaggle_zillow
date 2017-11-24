from data_exploration import Display_missing_percentages, random_forest_importance
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib as plt
from data_imputation import *
from sklearn.ensemble import ExtraTreesClassifier

DEBUG = 1


def feature_engineering(df_train, df_test):

    df_train, df_test = data_modification(df_train, df_test)
    df_train, df_test = missing_value_removal(df_train, df_test)
    df_train, df_test = new_features(df_train, df_test)
    df_train, df_test = missing_value_removal(df_train, df_test)
    # df_train, df_test = data_imputation(df_train,df_test)
    df_train, df_test = label_encoding(df_train, df_test)

    return df_train, df_test


def data_modification(df_train, df_test):
    print('Memory usage reduction...')
    df_train[['latitude', 'longitude']] /= 1e6
    df_test[['latitude', 'longitude']] /= 1e6

    df_train['censustractandblock'] /= 1e12
    df_test['censustractandblock'] /= 1e12

    return df_train,df_test


def missing_value_removal(df_train, df_test):
    # counting number of missing values
    cnt = Display_missing_percentages(df_train)

    if DEBUG:
        print 'Before Dropping Values'
        print df_train.shape
        print df_test.shape

    drop_list = []
    for c in df_train.columns:
        if cnt[c] > 90:
            if DEBUG:
                print c
                c123 = 0
            drop_list.append(c)

    df_train_new = df_train.drop(drop_list, axis=1)
    df_test_new = df_test.drop(drop_list, axis=1)

    df_train = df_train_new
    df_test = df_test_new

    if DEBUG:
        print 'After Dropping Values'
        print df_train.shape
        print df_test.shape

    return df_train, df_test


def label_encoding(df_train, df_test):
    # Label Encoding For Machine Learning &amp; Filling Missing Values
    #
    # We are now label encoding our datasets. All of the machine learning algorithms
    # employed in scikit learn assume that the data being fed to them is in numerical form.
    # LabelEncoding ensures that all of our categorical variables are in numerical representation.
    # Also note that we are filling the missing values in our dataset with a zero before label
    # encoding them.
    # This is to ensure that label encoder function does not experience any problems while
    # carrying out its operation #

    lbl = LabelEncoder()
    for c in df_train.columns:
        if df_train[c].dtype == 'object':
            lbl.fit(list(df_train[c].values))
            df_train[c] = lbl.transform(list(df_train[c].values))

    for c in df_test.columns:
        if df_test[c].dtype == 'object':
            lbl.fit(list(df_test[c].values))
            df_test[c] = lbl.transform(list(df_test[c].values))

    return df_train, df_test


def new_features(df_train, df_test):
    # living area proportions
    df_train['living_area_prop'] = df_train['calculatedfinishedsquarefeet'] / df_train['lotsizesquarefeet']
    df_test['living_area_prop'] = df_test['calculatedfinishedsquarefeet'] / df_test['lotsizesquarefeet']
    # tax value ratio
    df_train['value_ratio'] = df_train['taxvaluedollarcnt'] / df_train['taxamount']
    df_test['value_ratio'] = df_test['taxvaluedollarcnt'] / df_test['taxamount']
    # tax value proportions
    df_train['value_prop'] = df_train['structuretaxvaluedollarcnt'] / df_train[
        'landtaxvaluedollarcnt']  # built structure value / value of land
    df_test['value_prop'] = df_test['structuretaxvaluedollarcnt'] / df_test['landtaxvaluedollarcnt']

    df_train['home_age'] = 2016 - df_train['yearbuilt'];
    df_test['home_age'] = 2016 - df_test['yearbuilt'];

    return df_train, df_test


def feature_selection(df_train, df_test):
    ### Rearranging the DataSets ###

    # We will now drop the features that serve no useful purpose.
    # We will also split our data and divide it into the representation
    # to make it clear which features are to be treated as determinants
    # in predicting the outcome for our target feature.
    # Make sure to include the same features in the test set as were
    # included in the training set #

    # dropping properties which are not important or which are similar


    x_train = df_train.drop(
        ['parcelid', 'logerror', 'transactiondate', 'bathroomcnt', 'fips', 'pooltypeid7', 'calculatedbathnbr',
         'regionidcounty', 'threequarterbathnbr', 'assessmentyear', 'censustractandblock', 'yearbuilt',
         'regionidneighborhood', ], axis=1)

    x_test = df_test.drop(['parcelid', 'bathroomcnt', 'fips', 'pooltypeid7', 'calculatedbathnbr',
                           'regionidcounty', 'threequarterbathnbr', 'assessmentyear', 'censustractandblock',
                           'yearbuilt', 'regionidneighborhood', '201610', '201611',
                           '201612', '201710', '201711', '201712'], axis=1)

    x_train = x_train.values
    y_train = df_train['logerror'].values

    y_test = 0
    return x_train, y_train, x_test, y_test
