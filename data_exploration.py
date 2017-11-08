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
    name = 'visualze_'+featurename+'.png'
    plt.savefig(name)
    plt.show()


def Display_missing_percentages(train):
    cnt = {}
    for c in train.columns:
        k = train[c].isnull().sum()
        cnt[c] = float(k) / train.shape[0] * 100

    sorted_cnt = sorted(cnt.iteritems(), key=lambda (k, v): (v, k))
    freq = [k[1] for k in sorted_cnt]

    plt.figure(figsize=(22, 18))
    plt.barh(range(len(cnt)), freq, align="center")
    plt.yticks(range(len(cnt)), list(cnt.keys()))
    plt.xlabel('Percentage of missing values',fontsize=12)
    plt.title('Missing value % for each of feature',fontsize=12)
    plt.savefig('Missing_values.png')
    plt.show()
    return cnt