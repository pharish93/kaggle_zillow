import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()

def data_exploration(train_df):

    plt.figure(figsize=(8, 6))
    plt.scatter(range(train_df.shape[0]), np.sort(train_df.logerror.values))
    plt.xlabel('index', fontsize=12)
    plt.ylabel('logerror', fontsize=12)
    plt.savefig('log_error_scatter.png')
    plt.show()


    ulimit = np.percentile(train_df.logerror.values, 99)
    llimit = np.percentile(train_df.logerror.values, 1)
    train_df['logerror'].ix[train_df['logerror'] > ulimit] = ulimit
    train_df['logerror'].ix[train_df['logerror'] < llimit] = llimit

    plt.figure(figsize=(12, 8))
    sns.distplot(train_df.logerror.values, bins=50, kde=False)
    plt.xlabel('logerror', fontsize=12)
    plt.ylabel('Number of Properties',fontsize = 12)
    plt.title('Log Error Histogram plot of Train Data')
    plt.savefig('log_error_histogram.png')
    plt.show()

    train_df['transaction_month'] = train_df['transactiondate'].dt.month

    cnt_srs = train_df['transaction_month'].value_counts()
    plt.figure(figsize=(12, 6))
    sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[3])
    plt.xticks(rotation='vertical')
    plt.xlabel('Month of transaction', fontsize=12)
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.savefig('transaction_dates.png')
    plt.show()

    a  =10