import numpy as np

def calculate_r_squared(y_test, y_pred):
    # score = r2_score(y_test, y_pred)

    rss = np.sum((y_test - y_pred) ** 2)
    tss = np.sum((y_test - y_test.mean()) ** 2)
    score = rss/tss

    print score
