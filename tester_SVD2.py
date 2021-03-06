# -*- coding: UTF-8 -*-
import os
import timeit
import numpy as np
from openpyxl import Workbook
from collections import defaultdict
#from scikits.talkbox.features import mfcc 
from python_speech_features import mfcc

from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc
#from sklearn.cross_validation import ShuffleSplit
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from utils1 import GENRE_DIR, GENRE_LIST
import scipy
import scipy.io.wavfile
import matplotlib.pyplot as plt
from sklearn import datasets 
from sklearn.decomposition import TruncatedSVD 

# from utils import plot_roc, plot_confusion_matrix, GENRE_DIR, GENRE_LIST, TEST_DIR

# from ceps import read_ceps, create_ceps_test, read_ceps_test

from pydub import AudioSegment

genre_list = GENRE_LIST

clf = None

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#          Please run the classifier script first
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def create_fft(wavfile): 
    sample_rate, song_array = scipy.io.wavfile.read(wavfile)
    fft_features = abs(scipy.fft(song_array[:30000]))
    #print(song_array)
    base_fn, ext = os.path.splitext(wavfile)
    data_fn = base_fn + ".fft"
    np.save(data_fn, fft_features)
    #print data_fn
    return data_fn



def create_ceps_test(fn):
    """
        Creates the MFCC features from the test files,
        saves them to disk, and returns the saved file name.
    """
    sample_rate, X = scipy.io.wavfile.read(fn)
    # X[X==0]=1
    # np.nan_to_num(X)
    ceps= mfcc(X)
    bad_indices = np.where(np.isnan(ceps))
    b=np.where(np.isinf(ceps))
    ceps[bad_indices]=0
    ceps[b]=0
    base_fn, ext = os.path.splitext(fn)
    data_fn = base_fn + ".ceps"
    np.save(data_fn, ceps)
    #print "Written ", data_fn
    return data_fn


def read_fft(test_file):
    X = []
    y = []
    fft_features = np.load(test_file)
    X.append(fft_features)
    
    for label, genre in enumerate(genre_list):
        y.append(label)
    # for label, genre in enumerate(genre_list):
    #     # create UNIX pathnames to id FFT-files.
    #     genre_dir = os.path.join(base_dir, genre, "*.fft.npy")
    #     # get path names that math genre-dir
    #     file_list = glob.glob(genre_dir)
    #     for file in file_list:
    #         fft_features = np.load(file)
    #         X.append(fft_features)
    #         y.append(label)
    
    # print(X)
    # print(y)
    

    return np.array(X), np.array(y)


def read_ceps_test(test_file):
    """
        Reads the MFCC features from disk and
        returns them in a numpy array.
    """
    X = []
    y = []
    ceps = np.load(test_file)
    num_ceps = len(ceps)
    X.append(np.mean(ceps[int(num_ceps / 10):int(num_ceps * 9 / 10)], axis=0))
    for label, genre in enumerate(genre_list):
        y.append(label)
    return np.array(X), np.array(y)



def test_model_on_single_file(file_path):
    clf = joblib.load('saved_models/model_mfcc_knn_2.pkl')
    #clf = joblib.load('saved_models/model_mfcc_knn_fake2.pkl')
    #clf = joblib.load('saved_models/model_mfcc_knn.pkl')
    #clf = joblib.load('saved_models/model_fft_log.pkl')
    X, y = read_ceps_test(create_ceps_test(test_file)+".npy")
    #X,y=read_fft(create_fft(test_file)+".npy")
    #nsamples, nx, ny = X.shape
    # X = X.reshape((nsamples,nx*ny))
    # x=X[:30000]
    # print(x.shape)
    probs = clf.predict_proba(X)
    #print "\t".join(str(x) for x in genre_list)
    #print "\t".join(str("%.3f" % x) for x in probs[0])
    probs=probs[0]
    max_prob = max(probs)
    for i,j in enumerate(probs):
        if probs[i] == max_prob:
            max_prob_index=i
    
    #print max_prob_index
    predicted_genre = genre_list[max_prob_index]
    #print "\n\npredicted genre = ",predicted_genre
    dictionary = dict(zip(probs, genre_list))
    #print dictionary

    #for values in sorted(dictionary.iteritems(),reverse=True):
    #	print values

    #return predicted_genre
    return X,y
    #probs.sort(reverse=True)


if __name__ == "__main__":
	mypath = "test3"
	myfiles = os.listdir(mypath)
	print myfiles
	general_acc_num = 0
	queenloss_acc_num = 0
	virus_acc_num = 0
	# ?????????????????????????????????
	wb = Workbook()
	X_total=[]
	y_total=[]
	i=0
	# ??????????????????????????????
	ws = wb.active
	for f in myfiles:
		print f
		if f == 'general':
			general_path = "./test3/general/"
			general_list = os.listdir(general_path)
			total_general_num = len(general_list)
			general_w_queenloss_num = 0;
			general_w_virus_num = 0;
			for g in general_list:
				g2=g.split(".")
				if g2[1]!="wav":
					continue
				#print("general files:"+str(g))
				answer = "general"
				#print("answer:"+str(answer))
				test_file = os.path.join(general_path,g)
				X,y = test_model_on_single_file(test_file)
				np.array(X_total.extend(X))
				np.array(y_total.append(0))
				#print y_total
				#print("predict:"+str(predicted_genre))
				# Visualize result using SVD
		elif f == 'queenless':
			general_path = "./test3/queenless/"
			general_list = os.listdir(general_path)
			total_general_num = len(general_list)
			general_w_queenloss_num = 0;
			general_w_virus_num = 0;
			for g in general_list:
				g2=g.split(".")
				if g2[1]!="wav":
					continue
				#print("general files:"+str(g))
				answer = "general"
				#print("answer:"+str(answer))
				test_file = os.path.join(general_path,g)
				X,y = test_model_on_single_file(test_file)
				np.array(X_total.extend(X))				
				np.array(y_total.append(1))
				#print("predict:"+str(predicted_genre))
				# Visualize result using SVD
		elif f == 'virus':
			general_path = "./test3/virus/"
			general_list = os.listdir(general_path)
			total_general_num = len(general_list)
			general_w_queenloss_num = 0;
			general_w_virus_num = 0;
			for g in general_list:
				g2=g.split(".")
				if g2[1]!="wav":
					continue
				#print("general files:"+str(g))
				answer = "general"
				#print("answer:"+str(answer))
				test_file = os.path.join(general_path,g)
				X,y = test_model_on_single_file(test_file)
				np.array(X_total.extend(X))
				
				np.array(y_total.append(2))
				#print("predict:"+str(predicted_genre))
				# Visualize result using SVD
	#print X_total.shape
	#print y_total.shape
	svd = TruncatedSVD(n_components=2) 
	X_reduced = svd.fit_transform(X_total)
	# Initialize scatter plot with x and y axis values
	a=0
	b=0
	c=0
	for i in range(len(X_reduced)):
		if y_total[i] == 0:
			if a == 0:
				plt.scatter(X_reduced[i, 0], X_reduced[i, 1], c='green', s=25,label='normal')
				a=a+1
			else:
				plt.scatter(X_reduced[i, 0], X_reduced[i, 1], c='green', s=25)
		if y_total[i] == 1:
			if b == 0:
				plt.scatter(X_reduced[i, 0], X_reduced[i, 1], c='red', s=25,label='queenless')
				b=b+1
			else:
				plt.scatter(X_reduced[i, 0], X_reduced[i, 1], c='red', s=25)
		if y_total[i] == 2:
			if c == 0:
				plt.scatter(X_reduced[i, 0], X_reduced[i, 1], c='blue', s=25,label='virus')
				c=c+1
			else:
				plt.scatter(X_reduced[i, 0], X_reduced[i, 1], c='blue', s=25)
	
	plt.legend(loc='best')
	plt.savefig("iris_plot_test3_2.png")
	plt.show()
								
