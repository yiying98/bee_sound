# -*- coding: UTF-8 -*-

from itertools import cycle
import sklearn 
from sklearn import linear_model
from scipy import interp
from sklearn.metrics import accuracy_score
import scipy
import os
import sys
import glob
import numpy as np
from sklearn.externals import joblib




"""reads MFCC-files and prepares X_train and y_train.
genre_list must consist of names of folders/genres consisting of the required MFCC-files
base_dir must contain genre_list of directories
"""
def read_ceps(genre_list, base_dir):
	X= []
	y=[]
	for label, genre in enumerate(genre_list):
		for fn in glob.glob(os.path.join(base_dir, genre, "*.ceps.npy")):
			print fn
			ceps = np.load(fn)
			num_ceps = len(ceps)
			X.append(np.mean(ceps[int(num_ceps*1/10):int(num_ceps*9/10)], axis=0))
			#X.append(ceps)
			y.append(label)
	
	print(np.array(X).shape)
	print(len(y))
	return np.array(X), np.array(y)


def main():	

	#genre_list = ["classical", "jazz"] IF YOU WANT TO CLASSIFY ONLY CLASSICAL AND JAZZ

	#use FFT
	# X, y = read_fft(genre_list, base_dir_fft)
	# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .20)
	# print('\n******USING FFT******')
	# learn_and_classify(X_train, y_train, X_test, y_test, genre_list)
	# print('*********************\n')

	#use MFCC
	clf = joblib.load('saved_models/model_mfcc_log_1_without_knn.pkl')
	X= []
	ceps = np.load("2018_11_017_16_12_50.ceps.npy")
	num_ceps = len(ceps)
	X.append(np.mean(ceps[int(num_ceps*1/10):int(num_ceps*9/10)], axis=0))
	
	#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .20)
	#print("new1",X_train.shape)
	print('******USING MFCC******')
	knn_predictions = clf.predict(X)
	print knn_predictions
if __name__ == "__main__":
	main()
