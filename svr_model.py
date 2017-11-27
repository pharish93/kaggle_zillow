import numpy as np
import pandas as pd
from sklearn.svm import SVR

def svm_model_experiments(x_train,y_train,x_test):

	print 'SVR regression model'

	X = x_train.values
 	y = y_train.values

	model_svr_rbf = SVR(kernel='linear', C=1000, epsilon=0.1)
	model_svr_rbf.fit(X,y)
	print 'svm fitting done'
	Predicted_test_svr = model_svr_rbf.predict(x_test.values)

	return Predicted_test_svr
