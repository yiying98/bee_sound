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
##from sklearn.externals import joblib
import pickle
import joblib
from time import gmtime, strftime
import pymysql
import os
import datetime
from python_speech_features import mfcc
import scipy.io.wavfile
from utils1 import GENRE_DIR, GENRE_LIST
#import cPickle as pickle

genre_list = GENRE_LIST


"""reads MFCC-files and prepares X_train and y_train.
genre_list must consist of names of folders/genres consisting of the required MFCC-files
base_dir must contain genre_list of directories
"""
# Given a wavfile, computes mfcc and saves mfcc data
def create_ceps(wavfile):
	sampling_rate, song_array = scipy.io.wavfile.read(wavfile)
	#print(sampling_rate)
	"""Get MFCC
	ceps  : ndarray of MFCC
	mspec : ndarray of log-spectrum in the mel-domain
	spec  : spectrum magnitude
	"""
	ceps=mfcc(song_array)
	#ceps, mspec, spec= mfcc(song_array)
	#print(ceps.shape)
	#this is done in order to replace NaN and infinite value in array
	bad_indices = np.where(np.isnan(ceps))
	b=np.where(np.isinf(ceps))
	ceps[bad_indices]=0
	ceps[b]=0
	
	return write_ceps(ceps, wavfile)
	
# Saves mfcc data 
def write_ceps(ceps, wavfile):
	base_wav, ext = os.path.splitext(wavfile)
	data_wav = base_wav + ".ceps"
	ceps_result = np.save(data_wav, ceps)
	print (data_wav)
	
	return data_wav



def main(): 
	
	hive_id = 2
	data=[]
    #genre_list = ["classical", "jazz"] IF YOU WANT TO CLASSIFY ONLY CLASSICAL AND JAZZ

    #use FFT
    # X, y = read_fft(genre_list, base_dir_fft)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .20)
    # print('\n******USING FFT******')
    # learn_and_classify(X_train, y_train, X_test, y_test, genre_list)
    # print('*********************\n')

    #use MFCC
	clf = open('/media/pi/HV620S/KNN_log/saved_models/model_mfcc_LR_all_v1.pkl','rb+')
	clf_a = pickle.load(clf)
	
    ##    with open('saved_models/model_mfcc_log_1_without_knn.pkl','rb') as f:
    ##        print(f)
    ##        clf =pickle.load(f)
    #f=open('saved_models/model_mfcc_log_1_without_knn.pkl')
    #clf = pickle.load(f)
	X=[]
	
	path="/media/pi/HV620S/sound/" #待讀取的資料夾
	path_list=os.listdir(path)
	path_list.sort(reverse=True)
	n = 0
	for filename in path_list:
		if n == 1:
			target = filename
			#print target
			n += 1
			break
		else:
			n += 1
	
	
	result = create_ceps(path+target)
	
	#print result
	ceps = np.load(result+".npy")
	num_ceps = len(ceps)
	X.append(np.mean(ceps[int(num_ceps*1/10):int(num_ceps*9/10)], axis=0))
	#print('******USING MFCC******')
	knn_predictions = clf_a.predict(X)
	
	
	data.append(hive_id)
	data.append(knn_predictions)
	ISOTIMEFORMAT = '%Y-%m-%d %H:%M:%S'
	data.append(datetime.datetime.now().strftime(ISOTIMEFORMAT))
	print (data[0])
	print (data[1][0])
	print (data[2])
	DatabaseSender(data)
	
	
def DatabaseSender(data):
	db = pymysql.connect(host='140.112.94.59', user='root' , passwd='taipower', db='bee', port=33306)
	#db = pymysql.connect(host='localhost', user='root' , passwd='', db='bee')
	cursor = db.cursor()
	sql = "INSERT INTO `shenghao`(`id`, `hive_id`, `status`, `time`) VALUES (NULL, '"+str(data[0])+"', '"+str(data[1][0])+"', '"+str(data[2])+"')"
	#print sql
	#Execute the SQL command
	cursor.execute(sql)
	#print sql
	#Commit your changes in the database
	db.commit()
	#print 'Insert data successful...'
	#LogData('Insert Weather data successful...','weatherStatus.txt')
	db.close()
	
if __name__ == "__main__":
    main()
