import schedule  
import logging   
import time  
import pyaudio
import wave
import os
import re
import traceback
from schedule import Job, CancelJob, IntervalError  
from datetime import datetime, timedelta 


  
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',datefmt='%Y-%m-%d %H:%M',filename='python_record.log')

logger = logging.getLogger(os.path.basename(__file__))  
logger.setLevel(0)  


HOME = os.listdir("/home")[0]
hive_id = HOME

def get_device_id():
    p=pyaudio.PyAudio()
    for card in range(p.get_device_count()):
        if 'USB' in p.get_device_info_by_index(card).get('name'):
            device_id=re.search(r'hw:(\d*),',p.get_device_info_by_index(card).get('name')).group(1)
            #print(device_id)
            return int(device_id) 

def record():
    form_1 = pyaudio.paInt16
    chans = 1
    samp_rate = 44100 
    chunk = 4096
    record_secs = 60
    dev_index = get_device_id()
    print(dev_index)
    wav_output_filename = '/media/{HOME}/TOS/sound/'.format(HOME=HOME)+datetime.now().strftime("%Y%m%d-%H%M%S")+'.wav'
    audio = pyaudio.PyAudio()
    stream = audio.open(format = form_1,rate = samp_rate,channels = chans, \
                    input_device_index = dev_index,input = True, \
                    frames_per_buffer=chunk)
    logger.info('Record start at {}'.format(datetime.now())) 
    frames = []
    try:
        for ii in range(0,int((samp_rate/chunk)*record_secs)):
            data = stream.read(chunk,exception_on_overflow = False)
            frames.append(data)

        logger.info('Record end at {}'.format(datetime.now()))  

        stream.stop_stream()
        stream.close()
        audio.terminate()


        wavefile = wave.open(wav_output_filename,'wb')
        wavefile.setnchannels(chans)
        wavefile.setsampwidth(audio.get_sample_size(form_1))
        wavefile.setframerate(samp_rate)
        wavefile.writeframes(b''.join(frames))
        wavefile.close()
    except Exception:
        logging.error(traceback.format_exc())



def main():  
    #logging.info('start')
    #logging.error('error')
    schedule.every(70).seconds.do(record)
  
    while True:
        schedule.run_pending()
        time.sleep(1) 
  
  
if __name__ == '__main__':  
    main() 