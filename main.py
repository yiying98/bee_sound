# -*- coding: UTF-8 -*-
# This should be run under python2.7 !
import os
import sys
import pickle
import datetime
import traceback

import numpy as np
import scipy
import scipy.io.wavfile
import pymysql
from python_speech_features import mfcc

import utils

genre_list = utils.GENRE_LIST

# Given a wavfile, computes mfcc and saves mfcc data
def create_ceps(wav_file):
    sampling_rate, song_array = scipy.io.wavfile.read(wav_file)
    ceps = mfcc(song_array)
    ceps = np.nan_to_num(ceps)
    return ceps

def DatabaseSender(data):
    db = pymysql.connect(host='140.112.94.59', user='root',
                         passwd='taipower', db='110_bee_sound_test', port=33306)
    cursor = db.cursor()
    sql = "INSERT INTO `test`(`id`, `hive_id`, `status`, `time`) VALUES (NULL, '" + \
        str(data[0])+"', '"+str(data[1][0])+"', '"+str(data[2])+"')"

    print(sql) # TEST

    # TEST_NORMAL_OPEN
    try:
        cursor.execute(sql)
        db.commit()
        print('Insert data successful...')
    except Exception:
        print(traceback.format_exc())

    db.close()

def main():
    hive_id = os.listdir("/home")[0]

    wav_dir = "/media/{hive_id}/TOS/sound".format(hive_id=hive_id)
    wavs = [filename for filename in os.listdir(wav_dir) if ('wav' in filename)]
    wavs.sort(reverse=True)
    wav_filename = os.path.join(wav_dir, wavs[1])
    basename, ext = os.path.splitext(wav_filename)
    np.save(basename + '_ceps', create_ceps(wav_filename))

    ceps = np.load(basename + '_ceps.npy')

    # Use MFCC
    model = '/home/{hive_id}/bee_sound/saved_models/model_mfcc_LR_all_v1.pkl'.format(hive_id=hive_id)
    clf = open(model, 'rb+')
    clf_loaded = pickle.load(clf)

    num_ceps = len(ceps)

    X = []
    X.append(np.mean(ceps[int(num_ceps*1/10):int(num_ceps*9/10)], axis=0))
    knn_predictions = clf_loaded.predict(X)

    data = []
    data.append(hive_id)
    data.append(knn_predictions)
    ISOTIMEFORMAT = '%Y-%m-%d %H:%M:%S'
    data.append(datetime.datetime.now().strftime(ISOTIMEFORMAT))
    DatabaseSender(data)

if __name__ == "__main__":
    main()