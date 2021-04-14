# -*- coding: UTF-8 -*-

from itertools import cycle
import sklearn 
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
from scipy import interp
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import scipy
import os
import sys
import glob
import numpy as np
from utils1 import GENRE_DIR, GENRE_LIST
from sklearn.externals import joblib
from random import shuffle

"""reads FFT-files and prepares X_train and y_train.
genre_list must consist of names of folders/genres consisting of the required FFT-files
base_dir must contain genre_list of directories
"""
def read_fft(genre_list, base_dir):
	X = []
	y = []
	for label, genre in enumerate(genre_list):
		# create UNIX pathnames to id FFT-files.
		genre_dir = os.path.join(base_dir, genre, "*.fft.npy")
		# get path names that math genre-dir
		file_list = glob.glob(genre_dir)
		for file in file_list:
			fft_features = np.load(file)
			X.append(fft_features)
			y.append(label)
	
	return np.array(X), np.array(y)


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

def learn_and_classify(X_train, y_train, X_test, y_test, genre_list, n_classes):

	
	print(len(X_train))
	print(len(X_train[0]))

	#Logistic Regression classifier

	logistic_classifier = linear_model.logistic.LogisticRegression()
	logistic_classifier.fit(X_train, y_train)
	logistic_predictions = logistic_classifier.predict(X_test)
	logistic_accuracy = accuracy_score(y_test, logistic_predictions)
	logistic_cm = confusion_matrix(y_test, logistic_predictions)
	print("logistic accuracy = " + str(logistic_accuracy))
	print("logistic_cm:")
	print(logistic_cm)

	#change the pickle file when using another classifier eg model_mfcc_fft

	joblib.dump(logistic_classifier, 'saved_models/model_mfcc_log_9_without_log_ROC.pkl')

	#K-Nearest neighbour classifier

	knn_classifier = KNeighborsClassifier()
	
	knn_predictions = knn_classifier.fit(X_train, y_train).predict(X_test)
	print "+++++++++"
	print knn_predictions
	#knn_accuracy = accuracy_score(y_test, knn_predictions)
	#knn_cm = confusion_matrix(y_test, knn_predictions)
	joblib.dump(knn_classifier, 'saved_models/model_mfcc_log_9_without_knn_ROC.pkl')
	
	
	# Compute ROC curve and ROC area for each class
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	y_score = label_binarize(knn_predictions, classes=[0, 1])
	for i in range(n_classes):
		fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
		roc_auc[i] = auc(fpr[i], tpr[i])
	
	# Compute micro-average ROC curve and ROC area
	fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
	roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
	
	#print np.shape(fpr.shape)
	#print np.shape(tpr.shape)
	#print np.shape(roc_auc)
	
	plt.figure()
	lw = 2
	plt.plot(fpr[0], tpr[0], color='darkorange',
			 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	plt.legend(loc="lower right")
	plt.show()


def plot_confusion_matrix(cm, title, genre_list, cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(genre_list))
    plt.xticks(tick_marks, genre_list, rotation=45)
    plt.yticks(tick_marks, genre_list)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    #plt.figure(figsize=(300,300))
    plt.savefig(r"F:\final\PNG_save_ROC\1_knn_without_confusion.png")
    #plt.show()


def main():
	
	base_dir_fft  = GENRE_DIR
	base_dir_mfcc = GENRE_DIR
	
	"""list of genres (these must be folder names consisting .wav of respective genre in the base_dir)
	Change list if needed.
	"""
	genre_list = [ "normal","queenless","virus"]
	
	#genre_list = ["classical", "jazz"] IF YOU WANT TO CLASSIFY ONLY CLASSICAL AND JAZZ

	#use FFT
	# X, y = read_fft(genre_list, base_dir_fft)
	# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .20)
	# print('\n******USING FFT******')
	# learn_and_classify(X_train, y_train, X_test, y_test, genre_list)
	# print('*********************\n')

	#use MFCC
	clf = joblib.load('saved_models/model_mfcc_log_1_without_log.pkl')
	X,y= read_ceps(genre_list, base_dir_mfcc)
	y_roc = label_binarize(y, classes=[0, 1, 2])
	n_classes = y_roc.shape[1]
	#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .20)
	#print("new1",X_train.shape)
	print('******USING MFCC******')
	knn_predictions = clf.predict(X)
	#print "+++++++++"
	#print knn_predictions
	#knn_accuracy = accuracy_score(y_test, knn_predictions)
	#knn_cm = confusion_matrix(y_test, knn_predictions)
	#joblib.dump(knn_classifier, 'saved_models/model_mfcc_log_9_without_knn_ROC.pkl')
	
	
	# Compute ROC curve and ROC area for each class
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	y_score = label_binarize(knn_predictions, classes=[0, 1, 2])
	
	knn_accuracy = accuracy_score(y, knn_predictions)
	knn_cm = confusion_matrix(y, knn_predictions)
	plot_confusion_matrix(knn_cm, "Confusion matrix for MFCC classification", genre_list)
	
	for i in range(n_classes):
		fpr[i], tpr[i], _ = roc_curve(y_roc[:, i], y_score[:, i])
		roc_auc[i] = auc(fpr[i], tpr[i])
	
	# Compute micro-average ROC curve and ROC area
	fpr["micro"], tpr["micro"], _ = roc_curve(y_roc.ravel(), y_score.ravel())
	roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
	
	#print np.shape(fpr.shape)
	#print np.shape(tpr.shape)
	#print np.shape(roc_auc)
	
	plt.figure()
	lw = 2
	plt.plot(fpr[0], tpr[0], color='darkorange',
			 lw=lw, label='ROC curve (AUC = %0.2f)' % roc_auc[0])
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic')
	plt.legend(loc="lower right")
	plt.savefig(r"F:\final\PNG_save_ROC\1_knn_without_ROC.png")
	#plt.show()
	#learn_and_classify(X_train, y_train, X_test, y_test, genre_list, n_classes)
	print('*********************')
	

if __name__ == "__main__":
	main()
