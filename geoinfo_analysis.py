import numpy as np
import pandas as pd
import gc
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale=1.5)


from sklearn import neighbors
from sklearn.preprocessing import OneHotEncoder


from sklearn.preprocessing import LabelEncoder

#function to deal with variables that are actually string/categories
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

def fillna_knn(df, base, target, fraction=1, threshold=10, n_neighbors = 10):
    assert isinstance(base, list) or isinstance(base, np.ndarray) and isinstance(target, str)
    whole = [target] + base

    miss = df[target].isnull()
    notmiss = ~miss
    nummiss = miss.sum()

    enc = OneHotEncoder()
    X_target = df.loc[notmiss, whole].sample(frac=fraction)

    enc.fit(X_target[target].unique().reshape((-1, 1)))

    Y = enc.transform(X_target[target].values.reshape((-1, 1))).toarray()
    X = X_target[base]

    print('fitting')
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
    clf.fit(X, Y)

    print('the shape of active features: ', enc.active_features_.shape)

    print('perdicting')
    Z = clf.predict(df.loc[miss, base])

    numunperdicted = Z[:, 0].sum()
    if numunperdicted / nummiss * 100 < threshold:
        print('writing result to df')
        df.loc[miss, target] = np.dot(Z, enc.active_features_)
        print('num of unperdictable data: ', numunperdicted)
        return enc
    else:
        print('out of threshold: {}% > {}%'.format(numunperdicted / nummiss * 100, threshold))


def impute_geo_info(df_train,df_test):

    df_train.dropna(axis=0, subset=['latitude', 'longitude'], inplace=True)
    df_test.dropna(axis=0, subset=['latitude', 'longitude'], inplace=True)

    fillna_knn( df = df_train,
                  base = [ 'latitude', 'longitude' ] ,
                  target = 'regionidzip', fraction = 0.1 )

    fillna_knn(df=df_train,
               base=['latitude', 'longitude'],
               target='regionidcity', fraction=0.15)

    fillna_knn(df=df_train,
               base=['latitude', 'longitude'],
               target='lotsizesquarefeet', fraction=0.15, n_neighbors=1)

    fillna_knn(df=df_train,
               base=['latitude', 'longitude'],
               target='yearbuilt', fraction=0.15, n_neighbors=1)

    zoningcode2int(df=df_train,
                   target='propertyzoningdesc')

    fillna_knn(df=df_train,
               base=['latitude', 'longitude'],
               target='propertyzoningdesc', fraction=0.15, n_neighbors=1)

    zoningcode2int(df=df_train,
                   target='propertycountylandusecode')
    fillna_knn(df=df_train,
               base=['latitude', 'longitude'],
               target='propertycountylandusecode', fraction=0.15, n_neighbors=1)


    # test set

    fillna_knn( df = df_test,
                  base = [ 'latitude', 'longitude' ] ,
                  target = 'regionidzip', fraction = 0.1 )

    fillna_knn(df=df_test,
               base=['latitude', 'longitude'],
               target='regionidcity', fraction=0.15)

    fillna_knn(df=df_test,
               base=['latitude', 'longitude'],
               target='lotsizesquarefeet', fraction=0.15, n_neighbors=1)

    fillna_knn(df=df_test,
               base=['latitude', 'longitude'],
               target='yearbuilt', fraction=0.15, n_neighbors=1)

    zoningcode2int(df=df_test,
                   target='propertyzoningdesc')

    fillna_knn(df=df_test,
               base=['latitude', 'longitude'],
               target='propertyzoningdesc', fraction=0.15, n_neighbors=1)

    zoningcode2int(df=df_test,
                   target='propertycountylandusecode')
    fillna_knn(df=df_test,
               base=['latitude', 'longitude'],
               target='propertycountylandusecode', fraction=0.15, n_neighbors=1)


    return df_train,df_test