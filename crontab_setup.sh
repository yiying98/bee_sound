HIVE_ID=$(tail -n 1 /etc/passwd | awk -F':' '{print $1}')

sudo umount /dev/sda1
sudo ntfslabel /dev/sda1 TOS
cd /media/$HIVE_ID
sudo mkdir TOS
sudo chmod 777 /media/$HIVE_ID/TOS

MOUNT="@reboot root mount /dev/sda1 /media/$HIVE_ID/TOS"

RECORD_DATE="*/1 * * * * root sh date >> /home/$HIVE_ID/record.log"
RECORD="*/1 * * * * root sh /home/$HIVE_ID/Desktop/record.sh >> /home/$HIVE_ID/record.log 2>&1"
MAIN_DATE="*/1 * * * * root sh date >> /home/$HIVE_ID/main.log"
MAIN="*/1 * * * * $HIVE_ID python2 /home/$HIVE_ID/bee_sound/main.py >> /home/$HIVE_ID/main.log 2>&1"

sudo echo "0 0 * * * root reboot" >> /etc/crontab
sudo echo "$MOUNT" >> /etc/crontab
sudo echo "$RECORD_DATE" >> /etc/crontab
sudo echo "$RECORD" >> /etc/crontab
sudo echo "$MAIN_DATE" >> /etc/crontab
sudo echo "$MAIN" >> /etc/crontab