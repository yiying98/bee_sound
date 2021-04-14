import os
import timeit
import numpy as np
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
    clf = joblib.load('saved_models/model_mfcc_include_empty.pkl')
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

    return predicted_genre

    #probs.sort(reverse=True)


if __name__ == "__main__":

	#global traverse
	#for subdir, dirs, files in os.walk(GENRE_DIR):
	#    traverse = list(set(dirs).intersection(set(GENRE_LIST)))
	#    break

	#test_file = "/home/dhruvesh/Desktop/dsp-final/genres/blues/blues.00000.wav"
	mypath = "test3"
	myfiles = os.listdir(mypath)
	print myfiles
	general_acc_num = 0
	queenloss_acc_num = 0
	virus_acc_num = 0
	empty_acc_num = 0
	for f in myfiles:
		print f
		if f == 'general':
			general_path = "./test3/general/"
			general_list = os.listdir(general_path)
			total_general_num = len(general_list)
			general_w_queenloss_num = 0;
			general_w_virus_num = 0;
			general_w_empty_num = 0;
			for g in general_list:
				g2=g.split(".")
				if g2[1]!="wav":
					continue
				#print("general files:"+str(g))
				answer = "general"
				#print("answer:"+str(answer))
				test_file = os.path.join(general_path,g)
				predicted_genre = test_model_on_single_file(test_file)
				#print("predict:"+str(predicted_genre))
				
				if predicted_genre == answer:
					general_acc_num = general_acc_num+1
				elif predicted_genre == "queenloss":
					general_w_queenloss_num = general_w_queenloss_num+1
				elif predicted_genre == "virus":
					general_w_virus_num = general_w_virus_num+1
				elif predicted_genre == "empty":
					general_w_empty_num = general_w_empty_num+1
			general_acc = general_acc_num / total_general_num
			print("******************")
			print("general_acc_num:"+str(general_acc_num))
			print("general_w_queenloss_num:"+str(general_w_queenloss_num))
			print("general_w_virus_num:"+str(general_w_virus_num))
			print("general_w_empty_num:"+str(general_w_empty_num))
			#print("total_general_num:"+str(total_general_num))
			#print("General accuracy:"+str(general_acc))
			print("******************")
		elif f == 'queenloss':
			queenloss_path = "./test3/queenloss/"
			queenloss_list = os.listdir(queenloss_path)
			total_queenloss_num = len(queenloss_list)
			queenloss_w_general_num = 0;
			queenloss_w_virus_num = 0;
			queenloss_w_empty_num = 0;
			for q in queenloss_list:
				q2=q.split(".")
				if q2[1]!="wav":
					continue
				#print("queenloss files:"+str(q))
				answer = "queenloss"
				#print("answer:"+str(answer))
				test_file = os.path.join(queenloss_path,q)
				predicted_genre = test_model_on_single_file(test_file)
				#print("predict:"+str(predicted_genre))
				
				if predicted_genre == answer:
					queenloss_acc_num = queenloss_acc_num+1
				elif predicted_genre == "general":
					queenloss_w_general_num = queenloss_w_general_num+1
				elif predicted_genre == "virus":
					queenloss_w_virus_num = queenloss_w_virus_num+1
				elif predicted_genre == "empty":
					queenloss_w_empty_num = queenloss_w_empty_num+1
			queenloss_acc = queenloss_acc_num / total_queenloss_num
			print("******************")
			print("queenloss_acc_num:"+str(queenloss_acc_num))
			print("queenloss_w_general_num:"+str(queenloss_w_general_num))
			print("queenloss_w_virus_num:"+str(queenloss_w_virus_num))
			print("queenloss_w_empty_num:"+str(queenloss_w_empty_num))
			#print("total_queenloss_num:"+str(total_queenloss_num))
			#print("Queenloss accuracy:"+str(queenloss_acc))
			print("******************")
		elif f == 'virus':
			virus_path = "./test3/virus/"
			virus_list = os.listdir(virus_path)
			total_virus_num = len(virus_list)
			virus_w_general_num = 0;
			virus_w_queenloss_num = 0;
			virus_w_empty_num = 0;
			for v in virus_list:
				v2=v.split(".")
				if v2[1]!="wav":
					continue
				#print("virus files:"+str(v))
				answer = "virus"
				#print("answer:"+str(answer))
				test_file = os.path.join(virus_path,v)
				predicted_genre = test_model_on_single_file(test_file)
				#print("predict:"+str(predicted_genre))
				
				if predicted_genre == answer:
					virus_acc_num = virus_acc_num+1
				elif predicted_genre == "general":
					virus_w_general_num = virus_w_general_num+1
				elif predicted_genre == "queenloss":
					virus_w_queenloss_num = virus_w_queenloss_num+1
				elif predicted_genre == "empty":
					virus_w_empty_num = virus_w_empty_num+1
			virus_acc = virus_acc_num / total_virus_num
			print("******************")
			print("virus_acc_num:"+str(virus_acc_num))
			print("virus_w_general_num:"+str(virus_w_general_num))
			print("virus_w_queenloss_num:"+str(virus_w_queenloss_num))
			print("virus_w_empty_num:"+str(virus_w_empty_num))
			#print("total_virus_num:"+str(total_virus_num))
			#print("Virus accuracy:"+str(virus_acc))
			print("******************")
		else:
			empty_path = "./test3/empty/"
			empty_list = os.listdir(empty_path)
			total_empty_num = len(empty_list)
			empty_w_general_num = 0;
			empty_w_queenloss_num = 0;
			empty_w_virus_num = 0;
			for v in empty_list:
				v2=v.split(".")
				if v2[1]!="wav":
					continue
				#print("virus files:"+str(v))
				answer = "empty"
				#print("answer:"+str(answer))
				test_file = os.path.join(empty_path,v)
				predicted_genre = test_model_on_single_file(test_file)
				#print("predict:"+str(predicted_genre))
				
				if predicted_genre == answer:
					empty_acc_num = empty_acc_num+1
				elif predicted_genre == "general":
					empty_w_general_num = empty_w_general_num+1
				elif predicted_genre == "queenloss":
					empty_w_queenloss_num = empty_w_queenloss_num+1
				elif predicted_genre == "virus":
					empty_w_virus_num = empty_w_virus_num+1
			empty_acc = empty_acc_num / total_empty_num
			print("******************")
			print("empty_acc_num:"+str(empty_acc_num))
			print("empty_w_general_num:"+str(empty_w_general_num))
			print("empty_w_queenloss_num:"+str(empty_w_queenloss_num))
			print("empty_w_virus_num:"+str(empty_w_virus_num))
			#print("total_virus_num:"+str(total_virus_num))
			#print("Virus accuracy:"+str(virus_acc))
			print("******************")
			
	total_accuracy = (general_acc_num+queenloss_acc_num+virus_acc_num) / (total_general_num+total_queenloss_num+total_virus_num)
	print("total accuracy:"+str(total_accuracy))
				
		

	##test_file = r"test/virus/20190104-111201.wav"
	##test_file = r"test/virus/20190104-111201.wav"
	# nsamples, nx, ny = test_file.shape
	# test_file = test_file.reshape((nsamples,nx*ny))
	# should predict genre as "ROCK"
	##predicted_genre = test_model_on_single_file(test_file)
	