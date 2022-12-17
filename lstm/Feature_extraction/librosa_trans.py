import librosa
import traceback

def log_mel_spec_transform(file:str,mfcc_max_padding=0,n_fft=2048, hop_length=512, n_mels=128):
    try:
        mel_in,sr = librosa.load(file)
        nor_mel_in = librosa.util.normalize(mel_in)
        mel_spec = librosa.feature.melspctrogram(nor_mel_in,sr,n_fft=2048, hop_length=512, n_mels=128)
        mel_db =  librosa.power_to_db(abs(mel_spec))
        normalized_mel = librosa.util.normalize(mel_db)
        shape = normalized_mel.shape[1]
        if (mfcc_max_padding > 0 & shape < mfcc_max_padding):
                xDiff = mfcc_max_padding - shape
                xLeft = xDiff//2
                xRight = xDiff-xLeft
                normalized_mel = np.pad(normalized_mel, pad_width=((0,0), (xLeft, xRight)), mode='constant')
        return normalized_mel
                
    except Exception as e:
        print("Error parsing wavefile: ", traceback.format_exc(e))
        return None 


def mfcc_transform(file:str, mfcc_max_padding=0, n_mfcc=40):
    try:
        mel_in,sr= librosa.load(file)
        nor_mal_in = librosa.util.normalize(mel_in)
        mfccs = librosa.feature.mfcc(y=mel_in,sr=sr,n_mfcc=40)
        normalized_mfcc = librosa.util.normalize(mfccs)
        shape = normalized_mfcc.shape[1]
        if (mfcc_max_padding > 0 & shape < mfcc_max_padding):
                xDiff = mfcc_max_padding - shape
                xLeft = xDiff//2
                xRight = xDiff-xLeft
                normalized_mfcc = np.pad(normalized_mfcc, pad_width=((0,0), (xLeft, xRight)), mode='constant')
        return normalized_mfcc
    except Exception as e:
        print("Error parsing wavefile: ",traceback.format_exc(e))
        return None 
    



if __name__ == '__main__':
        print(mfcc_transform('/Volumes/TOSHIBA EXT/old/test/normal/2018_05_016_0_0_59.wav'))
