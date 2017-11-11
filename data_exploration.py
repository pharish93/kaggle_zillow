import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()

def data_exploration(train_df):

    plt.figure(figsize=(8, 6))
    plt.scatter(range(train_df.shape[0]), np.sort(train_df.logerror.values))
    plt.xlabel('index', fontsize=16)
    plt.ylabel('logerror', fontsize=16)
    plt.savefig('./images/log_error_scatter.png')
    plt.show()


    ulimit = np.percentile(train_df.logerror.values, 99)
    llimit = np.percentile(train_df.logerror.values, 1)
    train_df['logerror'].ix[train_df['logerror'] > ulimit] = ulimit
    train_df['logerror'].ix[train_df['logerror'] < llimit] = llimit

    plt.figure(figsize=(12, 8))
    sns.distplot(train_df.logerror.values, bins=50, kde=False)
    plt.xlabel('logerror', fontsize=16)
    plt.ylabel('Number of Properties',fontsize = 16)
    plt.title('Log Error Histogram plot of Train Data',fontsize=16)
    plt.savefig('./images/log_error_histogram.png')
    plt.show()

    train_df['transaction_month'] = train_df['transactiondate'].dt.month

    cnt_srs = train_df['transaction_month'].value_counts()
    plt.figure(figsize=(16, 6))
    sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[3])
    plt.xticks(rotation='vertical')
    plt.xlabel('Month of transaction', fontsize=16)
    plt.ylabel('Number of Occurrences', fontsize=16)
    plt.title('Transaction occurences by month',fontsize=16)
    plt.savefig('./images/transaction_dates.png')
    plt.show()

    train_df.drop(['transaction_month'],axis=1)
    a  =10

def visualize_distribution(properties,df_train,featurename):
    mean_c = df_train[featurename].mean()
    df_train[featurename] = df_train[featurename].fillna(mean_c)
    properties[featurename]=properties[featurename].fillna(mean_c)


    ulimit = np.percentile(df_train[featurename], 99)
    llimit = np.percentile(df_train[featurename], 1)
    df_train[featurename].ix[df_train[featurename] > ulimit] = ulimit
    df_train[featurename].ix[df_train[featurename] < llimit] = llimit

    properties[featurename].ix[properties[featurename] > ulimit] = ulimit
    properties[featurename].ix[properties[featurename] < llimit] = llimit


    plt.figure(figsize=(16, 8))
    sns.kdeplot(df_train[featurename],shade=True,label = 'Train plot')
    sns.kdeplot(properties[featurename], shade=True, label='Properties plot')
    plt.xlabel(featurename, fontsize=16)
    plt.ylabel('Density', fontsize=16)
    plt.title('Distribution in Properties vs Train Samples',fontsize=16)
    name = './images/visualze_'+featurename+'.png'
    plt.savefig(name)
    plt.show()


# def Display_missing_percentages_old(train):
#     cnt = {}
#     cnt_full = {}
#     for c in train.columns:
#         k = train[c].isnull().sum(axis=0)
#
#         cnt[c] = (float(k) / train.shape[0]) * 100
#         cnt_full[c] = float(k)
#         print c,cnt[c],cnt_full[c], "\n"
#
#     sorted_cnt = sorted(cnt.iteritems(), key=lambda (k, v): (v, k))
#
#     freq = [k[1] for k in sorted_cnt]
#
#     plt.figure(figsize=(22, 18))
#     plt.barh(range(len(cnt)), freq, align="center")
#     plt.yticks(range(len(cnt)), list(cnt.keys()))
#     plt.xlabel('Percentage of missing values',fontsize=12)
#     plt.title('Missing value % for each of feature',fontsize=12)
#     plt.savefig('./images/Missing_values.png')
#     plt.show()
#     return cnt


def Display_missing_percentages(prop_df):
    missing_df = prop_df.isnull().sum(axis=0).reset_index()
    missing_df.columns = ['column_name', 'missing_count']
    missing_df = missing_df.ix[missing_df['missing_count']>0]
    missing_df = missing_df.sort_values(by='missing_count')

    ind = np.arange(missing_df.shape[0])
    width = 0.9
    fig, ax = plt.subplots(figsize=(12,18))
    rects = ax.barh(ind, missing_df.missing_count.values, color='blue')
    ax.set_yticks(ind)
    ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
    ax.set_xlabel("Count of missing values")
    ax.set_title("Number of missing values in each column")
    plt.savefig('./images/Missing_values.png')
    plt.show()

from sklearn.ensemble import RandomForestRegressor
def random_forest_importance(df_train):
     # Build a forest and compute the feature importances
    forest = RandomForestRegressor(n_estimators=250)
    y = df_train['logerror'].values.ravel()

    x_try = df_train.columns[:-1]
    X1 = df_train.drop(['parcelid', 'logerror', 'transactiondate'], axis=1)

    X = X1[X1.columns[:-1]].values
    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    indices = np.flipud(indices)

    # Print the feature ranking
    print("Feature ranking:")
    col_names = np.array([])
    for f in range(X.shape[1]):
        col_names = np.append(col_names,X1.columns[indices[f]])

    # Plot the feature importances of the forest
    plt.rcParams.update({'font.size': 7})
    plt.figure()
    plt.title("Feature importances")
    plt.xlabel("Importance")
    plt.barh(range(X.shape[1]), importances[indices],
            color="b", yerr=std[indices], align="center")
    plt.yticks(range(X.shape[1]), col_names)
    plt.ylim([-1, X.shape[1]])
    plt.show()
    plt.savefig('./images/Feature_importance.png')
