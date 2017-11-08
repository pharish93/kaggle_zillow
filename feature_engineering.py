from data_exploration import Display_missing_percentages
from sklearn.preprocessing import LabelEncoder

DEBUG = 1
def data_preprocessing(df_train,df_test):

    # # living area proportions
    # df_train['living_area_prop'] = df_train['calculatedfinishedsquarefeet'] / df_train['lotsizesquarefeet']
    # df_test['living_area_prop'] = df_test['calculatedfinishedsquarefeet'] / df_test['lotsizesquarefeet']
    # # tax value ratio
    # df_train['value_ratio'] = df_train['taxvaluedollarcnt'] / df_train['taxamount']
    # df_test['value_ratio'] = df_test['taxvaluedollarcnt'] / df_test['taxamount']
    # # tax value proportions
    # df_train['value_prop'] = df_train['structuretaxvaluedollarcnt'] / df_train['landtaxvaluedollarcnt']  # built structure value / value of land
    # df_test['value_prop'] = df_test['structuretaxvaluedollarcnt'] / df_test['landtaxvaluedollarcnt']
    #

    print('Memory usage reduction...')
    df_train[['latitude', 'longitude']] /= 1e6
    df_test[['latitude', 'longitude']] /= 1e6

    df_train['censustractandblock'] /= 1e12
    df_test['censustractandblock'] /= 1e12

    # counting number of missing values
    cnt = Display_missing_percentages(df_train)

    if DEBUG :
        print 'Before Dropping Values'
        print df_train.shape
        print df_test.shape

    drop_list = []
    for c in df_train.columns:
        if cnt[c] > 90:
            if DEBUG :
                print c
            drop_list.append(c)

    df_train_new = df_train.drop(drop_list,axis=1)
    drop_list.extend(('201610', '201611','201612', '201710', '201711', '201712'))
    df_test_new = df_test.drop(drop_list,axis=1)


    df_train = df_train_new
    df_test = df_test_new

    if DEBUG :
        print 'After Dropping Values'
        print df_train.shape
        print df_test.shape

    # cnt_new = Display_missing_percentages(df_train)

    # random_forest_importance(df_train)
    df_train,df_test=label_encoding(df_train,df_test)
    return df_train,df_test

from sklearn.ensemble import ExtraTreesClassifier
def random_forest_importance(df_train):
    # Build a forest and compute the feature importances
    forest = ExtraTreesClassifier(n_estimators=250,
                                  random_state=0)
    y = df_train['logerror']
    x_try = df_train.columns[:-1]
    X = df_train.drop(['parcelid', 'logerror', 'transactiondate' ], axis=1)
    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.show()


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

    ignore_labels = ['parcelid','transactiondata','logerror']
    lbl = LabelEncoder()
    for c in df_train.columns:

        if (c != 'parcelid' and c!= 'transactiondate' and c !='logerror'):
            if df_train[c].dtype == 'object':
                df_train[c] = df_train[c].fillna(0)
            else:
                mean_c = df_train[c].mean()
                df_train[c]=df_train[c].fillna(mean_c)

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


    k = ['basementsqft','bathroomcnt','censustractandblock']

    x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc',
                             'propertycountylandusecode', ], axis=1)

    x_test = df_test.drop(['parcelid', 'propertyzoningdesc',
                           'propertycountylandusecode', '201610', '201611',
                           '201612', '201710', '201711', '201712'], axis = 1)

    x_train = x_train.values
    y_train = df_train['logerror'].values

    y_test = 0
    return x_train,y_train,x_test,y_test
