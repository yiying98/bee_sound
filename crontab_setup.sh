HIVE_ID=$(tail -n 1 /etc/passwd | awk -F':' '{print $1}')

sudo umount /dev/sda1
sudo ntfslabel /dev/sda1 TOS
cd /media/$HIVE_ID
sudo mkdir TOS
sudo chmod 777 /media/$HIVE_ID/TOS

RECORD="* * * * * root sh /home/$HIVE_ID/bee_sound/record.sh >> /home/$HIVE_ID/Desktop/record.log 2>&1"
MAIN="* * * * * $HIVE_ID sh /home/$HIVE_ID/bee_sound/main.sh >> /home/$HIVE_ID/Desktop/main.log 2>&1"

sudo echo "0 0 * * * root reboot" >> /etc/crontab
sudo echo "@reboot root mount /dev/sda1 /media/$HIVE_ID/TOS" >> /etc/crontab
sudo echo "$RECORD" >> /etc/crontab
sudo echo "$MAIN" >> /etc/crontab