HIVE_ID=$(tail -n 1 /etc/passwd | awk -F':' '{print $1}')

sudo umount /dev/sda1
sudo ntfslabel /dev/sda1 TOS
cd /media/$HIVE_ID
sudo mkdir TOS
sudo chmod 777 /media/$HIVE_ID/TOS

RECORD="@reboot $HIVE_ID python /home/$HIVE_ID/bee_sound/python_record.py"
MAIN="@reboot $HIVE_ID python /home/$HIVE_ID/bee_sound/main.py"

sudo echo "0 0 * * * root reboot" >> /etc/crontab
sudo echo "@reboot root mount /dev/sda1 /media/$HIVE_ID/TOS" >> /etc/crontab
sudo echo "$RECORD" >> /etc/crontab
sudo echo "$MAIN" >> /etc/crontab

sudo mount /dev/sda1 /media/$HIVE_ID/TOS
