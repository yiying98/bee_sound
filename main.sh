date +'%F %T'
HIVE_ID=$(tail -n 1 /etc/passwd | awk -F':' '{print $1}')
python2 /home/$HIVE_ID/bee_sound/main.py
printf '\n'
exit 0