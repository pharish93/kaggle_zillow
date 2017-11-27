import numpy as np
from sklearn.metrics import mean_absolute_error

def error_metric_calc( y_true,y_pred):

    calculate_r_squared(y_true,y_pred)
    mean_abs_error(y_true,y_pred)


def calculate_r_squared(y_true, y_pred):
    # score = r2_score(y_true, y_pred)

    rss = np.sum((y_true - y_pred) ** 2)
    tss = np.sum((y_true - y_true.mean()) ** 2)
    score = rss/tss
    print score

def mean_abs_error(y_true, y_pred):

    score = mean_absolute_error(y_true, y_pred)

    print score
