# get hive_id from username
HIVE_ID=$(tail -n 1 /etc/passwd | awk -F':' '{print $1}')

# hdd setup
sudo umount /dev/sda1
sudo ntfslabel /dev/sda1 TOS
cd /media/$HIVE_ID
sudo mkdir TOS
cd /media/$HIVE_ID/TOS
sudo mkdir sound
sudo chmod 777 /media/$HIVE_ID/TOS
sudo mount /dev/sda1 /media/$HIVE_ID/TOS
cd /home/$HIVE_ID

# crontab setup
RECORD="@reboot $HIVE_ID python /home/$HIVE_ID/bee_sound/python_record.py"
MAIN="@reboot $HIVE_ID python /home/$HIVE_ID/bee_sound/main.py"

sudo echo "0 0 * * * root reboot" >> /etc/crontab
sudo echo "@reboot root mount /dev/sda1 /media/$HIVE_ID/TOS" >> /etc/crontab
sudo echo "$RECORD" >> /etc/crontab
sudo echo "$MAIN" >> /etc/crontab

# python modules setup
sudo apt update
sudo apt upgrade
sudo apt install gfortran libatlas-base liblapack-dev
sudo apt install numpy
sudo apt install scipy
sudo apt install python-sklearn python-sklearn-lib python-sklearn-doc
sudo apt install python-mysqldb
sudo git clone http://people.csail.mit.edu/hubert/git/pyaudio.git
cd /home/$HIVE_ID
sudo apt-get install libportaudio0 libportaudio2 libportaudiocpp0 portaudio19-dev
sudo apt-get install python-dev
cd pyaudio
sudo python setup.py install