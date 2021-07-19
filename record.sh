date +'%F %T'
HIVE_ID=$(tail -n 1 /etc/passwd | awk -F':' '{print $1}')
sudo arecord -D plughw:2,0 -f S16_LE -d 60 -r 44100 --use-strftime /media/$HIVE_ID/TOS/sound/%Y%m%d-%H%M%v.wav
sudo pkill -9 arecord
chmod -R 777 /media/$HIVE_ID/TOS/sound
printf '\n'
exit 0