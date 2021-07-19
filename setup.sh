#!/bin/shell

sudo apt update
sudo apt upgrade
sudo apt install gfortran libatlas-base liblapack-dev
sudo apt install numpy
sudo apt install scipy
sudo apt install python-sklearn python-sklearn-lib python-sklearn-doc
sudo apt install pymysql
pip install schedule
sh -x audio_setup.sh
sh -x crontab_setup.sh