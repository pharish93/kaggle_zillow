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
import random
import datetime as dt
import gc
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


def load_full_data():

    # Load the Datasets #
    # We need to load the datasets that will be needed to train our machine learning algorithms,
    #  handle our data and make predictions.
    # Note that these datasets are the ones that are already provided once you enter the
    # competition by accepting terms and conditions

    train = pd.read_csv('./data/train_2016_v2.csv', parse_dates=["transactiondate"])
    properties = pd.read_csv('./data/properties_2016.csv')
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

def data_preprocessing(df_train,df_test):

    # living area proportions
    df_train['living_area_prop'] = df_train['calculatedfinishedsquarefeet'] / df_train['lotsizesquarefeet']
    df_test['living_area_prop'] = df_test['calculatedfinishedsquarefeet'] / df_test['lotsizesquarefeet']
    # tax value ratio
    df_train['value_ratio'] = df_train['taxvaluedollarcnt'] / df_train['taxamount']
    df_test['value_ratio'] = df_test['taxvaluedollarcnt'] / df_test['taxamount']
    # tax value proportions
    df_train['value_prop'] = df_train['structuretaxvaluedollarcnt'] / df_train['landtaxvaluedollarcnt']  # built structure value / value of land
    df_test['value_prop'] = df_test['structuretaxvaluedollarcnt'] / df_test['landtaxvaluedollarcnt']


    print('Memory usage reduction...')
    df_train[['latitude', 'longitude']] /= 1e6
    df_test[['latitude', 'longitude']] /= 1e6

    df_train['censustractandblock'] /= 1e12
    df_test['censustractandblock'] /= 1e12

    # counting number of missing values

    cnt = {}
    for c in df_train.columns:
        k = df_train[c].isnull().sum()
        cnt[c] = float( k) / df_train.shape[0] * 100


    sorted_cnt = sorted(cnt.iteritems(), key=lambda (k, v): (v, k))
    freq = [ k[1] for k in sorted_cnt]

    plt.bar(range(len(cnt)), freq, align="center")
    plt.xticks(range(len(cnt)), list(cnt.keys()))
    plt.show()


    df_train,df_test=label_encoding(df_train,df_test)
    return df_train,df_test



def label_encoding(df_train,df_test):

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
        df_train[c]=df_train[c].fillna(0)
        if df_train[c].dtype == 'object':
            lbl.fit(list(df_train[c].values))
            df_train[c] = lbl.transform(list(df_train[c].values))

    for c in df_test.columns:
        df_test[c]=df_test[c].fillna(0)
        if df_test[c].dtype == 'object':
            lbl.fit(list(df_test[c].values))
            df_test[c] = lbl.transform(list(df_test[c].values))

    return df_train,df_test

def feature_selection(df_train,df_test):

    ### Rearranging the DataSets ###

    # We will now drop the features that serve no useful purpose.
    # We will also split our data and divide it into the representation
    # to make it clear which features are to be treated as determinants
    # in predicting the outcome for our target feature.
    # Make sure to include the same features in the test set as were
    # included in the training set #


    x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc',
                             'propertycountylandusecode', ], axis=1)

    x_test = df_test.drop(['parcelid', 'propertyzoningdesc',
                           'propertycountylandusecode', '201610', '201611',
                           '201612', '201710', '201711', '201712'], axis = 1)

    x_train = x_train.values
    y_train = df_train['logerror'].values

    y_test = 0
    return x_train,y_train,x_test,y_test

def model_experiments(x_train,y_train,x_test):
    ### Cross Validation ###

    # We are dividing our datasets into the training and validation sets so that
    # we could monitor and the test the progress of our machine learning algorithm.
    # This would let us know when our model might be over or under fitting on the
    # dataset that we have employed. #

    from sklearn.model_selection import train_test_split

    X = x_train
    y = y_train

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

    # Submitting the Results
    # Once again load the file and start submitting the results in each column

    sample_file = pd.read_csv('./data/sample_submission.csv')
    for c in sample_file.columns[sample_file.columns != 'ParcelId']:
        sample_file[c] = Predicted_test_xgb

    print('Preparing the csv file ...')
    sample_file.to_csv('xgb_predicted_results.csv', index=False, float_format='%.4f')
    print("Finished writing the file")



def main():
    # df_train,df_test = load_full_data()
    df_train, df_test = load_small_data()
    df_train,df_test = data_preprocessing(df_train,df_test)

    x_train, y_train, x_test,k = feature_selection(df_train,df_test)
    model_experiments(x_train,y_train,x_test)



if __name__ == "__main__":
    main()
