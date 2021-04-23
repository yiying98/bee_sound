# -*- coding: utf-8 -*-
"""
Created on 2021/03/30
@author: Jim
"""
import ftplib    
import os
import time
import MySQLdb
from datetime import datetime
camera_ID = 'Yunlin-1'
receivetime = time.strftime('%m%d_%H%M',time.localtime(time.time()))
receivetime_SQL = datetime.now()
os.system('raspistill -o /home/pi/Desktop/'+camera_ID+'.jpg -t 3000')
a=time.time()
def CPU_temperature():
   global CPU_temp
   file = open("/sys/class/thermal/thermal_zone0/temp")
   # 读取结果，并转换为浮点数
   CPU_temp_float = float(file.read()) / 1000
   # 关闭文件
   file.close()
   # 向控制台打印
   #print "temp : %3f" %CPU_temp
   CPU_temp = str(CPU_temp_float)
   print 'CPU_temp:' + CPU_temp
 

try:

    #FTP 連線 61備份
    ftp = ftplib.FTP()   #創ftp對象 
    FTPIP= "140.112.94.61"  
    FTPPORT= 3837  
    USERNAME= "r09631029"  
    USERPWD= "@r09631029"  
    ftp.connect(FTPIP, FTPPORT)  #連結  
    ftp.login(USERNAME,USERPWD)  #登入帳密  
    print("[FTP] Login...")
    print(ftp.getwelcome()) 
    bufsize = 1024
    file_handler = open(("/home/pi/Desktop/"+camera_ID+".jpg"),'rb')
    ftp.cwd('/Lab303/01_研究計畫案/110_防檢局_利用自動化監測建立果實蠅非疫生產點/溫室進出入影像/雲林斗六')
    ftp.storbinary('STOR %s' % os.q`path.basename(camera_ID+'_%s.jpg'%receivetime),file_handler,bufsize)
    ftp.set_debuglevel(0) 
    file_handler.close() 
    ftp.quit() 
    print ("ftp_61 upload OK")

    
except Exception as E:
    print (E)
    if os.path.exists("unsend") == False: os.mkdir("unsend")
    os.system(('cp /home/pi/Desktop/'+camera_ID+'.jpg /home/pi/Desktop/unsend/'+camera_ID+'_'+receivetime+'.jpg'))

print(time.time()-a)

