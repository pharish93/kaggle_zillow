import numpy as np
import pandas as pd
import matplotlib as plt
from geoinfo_analysis import *

def zoningcode2int( df, target ):
    storenull = df[ target ].isnull()
    enc = LabelEncoder( )
    df[ target ] = df[ target ].astype( str )

    print('fit and transform')
    df[ target ]= enc.fit_transform( df[ target ].values )
    print( 'num of categories: ', enc.classes_.shape  )
    df.loc[ storenull, target ] = np.nan
    print('recover the nan value')
    return enc

def data_imputation(df_train, df_test):
    # imputing missing values with corresponding data to be filled

    df_train, df_test = impute_floors(df_train, df_test)
    df_train,df_test = impute_geo_info(df_train,df_test)
    df_train, df_test = impute_bathroom(df_train, df_test)

    mode_imputations = ['airconditioningtypeid', 'heatingorsystemtypeid', 'fireplacecnt', 'garagecarcnt', 'roomcnt',
                        'bedroomcnt', 'propertycountylandusecode','propertylandusetypeid']
    for element in mode_imputations:
        mode_element = df_train[element].mode()
        df_test[element] = df_test[element].fillna(mode_element[0])
        df_train[element] = df_train[element].fillna(mode_element[0])

    zero_imputations = ['poolcnt']
    for element in zero_imputations:
        df_test[element] = df_test[element].fillna(0)
        df_train[element] = df_train[element].fillna(0)


    mean_imputations = ['fullbathcnt','calculatedfinishedsquarefeet','latitude','longitude','lotsizesquarefeet']
    for element in mean_imputations:
        mean_element = df_train[element].mean()
        df_test[element] = df_test[element].fillna(mean_element)
        df_train[element] = df_train[element].fillna(mean_element)

    mean_round = ['buildingqualitytypeid', 'taxvaluedollarcnt','structuretaxvaluedollarcnt','landtaxvaluedollarcnt','taxamount']
    for element in mean_round:
        mean_element = round(df_train[element].mean())
        df_test[element] = df_test[element].fillna(mean_element)
        df_train[element] = df_train[element].fillna(mean_element)



    return df_train, df_test


def impute_bathroom(df_train, df_test):
    loc = df_train['fullbathcnt'].index[df_train['fullbathcnt'].apply(np.isnan)]
    df_train['fullbathcnt'][loc] = df_train['bathroomcnt'][loc]

    return df_train, df_test


def impute_floors(df_train, df_test):
    # Room imputations
    loc = df_train['unitcnt'].index[df_train['unitcnt'].apply(np.isnan)]
    df_train['unitcnt'][loc] = df_train['numberofstories'][loc]
    loc = df_test['unitcnt'].index[df_test['unitcnt'].apply(np.isnan)]
    df_test['unitcnt'][loc] = df_test['numberofstories'][loc]

    # Train
    loc = df_train['unitcnt'].index[df_train['unitcnt'].apply(np.isnan)].tolist()
    list_intersect = df_train.index[df_train['propertylandusetypeid'] == 246].tolist()
    common1 = list(set(loc).intersection(list_intersect))
    df_train['unitcnt'][common1] = 2

    loc = df_train['unitcnt'].index[df_train['unitcnt'].apply(np.isnan)].tolist()
    list_intersect = df_train.index[df_train['propertylandusetypeid'] == 247].tolist()
    common1 = list(set(loc).intersection(list_intersect))
    df_train['unitcnt'][common1] = 3

    loc = df_train['unitcnt'].index[df_train['unitcnt'].apply(np.isnan)].tolist()
    list_intersect = df_train.index[df_train['propertylandusetypeid'] == 248].tolist()
    common1 = list(set(loc).intersection(list_intersect))
    df_train['unitcnt'][common1] = 4
    loc = df_train['unitcnt'].index[df_train['unitcnt'].apply(np.isnan)]

    df_train['unitcnt'][loc] = 1

    # Test
    loc = df_test['unitcnt'].index[df_test['unitcnt'].apply(np.isnan)].tolist()
    list_intersect = df_test.index[df_test['propertylandusetypeid'] == 246].tolist()
    common1 = list(set(loc).intersection(list_intersect))
    df_test['unitcnt'][common1] = 2

    loc = df_test['unitcnt'].index[df_test['unitcnt'].apply(np.isnan)].tolist()
    list_intersect = df_test.index[df_test['propertylandusetypeid'] == 247].tolist()
    common1 = list(set(loc).intersection(list_intersect))
    df_test['unitcnt'][common1] = 3

    loc = df_test['unitcnt'].index[df_test['unitcnt'].apply(np.isnan)].tolist()
    list_intersect = df_test.index[df_test['propertylandusetypeid'] == 248].tolist()
    common1 = list(set(loc).intersection(list_intersect))
    df_test['unitcnt'][common1] = 4

    loc = df_test['unitcnt'].index[df_test['unitcnt'].apply(np.isnan)]

    df_test['unitcnt'][loc] = 1

    return df_train, df_test
