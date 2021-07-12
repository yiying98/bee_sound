echo data
sudo arecord  -D plughw:2,0 -f S16_LE -d 10 -r 44100 --use-strftime /media/normal/TOSHIBA\ EXT/sound/%Y%m%d-%H%M%v.wav
