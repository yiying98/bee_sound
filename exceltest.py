# -*- coding: UTF-8 -*-
from openpyxl import Workbook
import os

# 創建一個空白活頁簿物件
wb = Workbook()

# 選取正在工作中的表單
ws = wb.active

mypath = "test2"
myfiles = os.listdir(mypath)
# 創建一個空白活頁簿物件
wb = Workbook()

# 選取正在工作中的表單
ws = wb.active

for f in myfiles:
	if f == 'general':
		general_path = "./test2/general/"
		general_list = os.listdir(general_path)
		total_general_num = len(general_list)
		for g in general_list:
			g2=g.split(".")
			if g2[1]!="wav":
				continue
			g3=g2[0].split("_")
			g4=g3[0]+"/"+g3[1]+"/"+g3[2]+" "+g3[3]+":"+g3[4]+":"+g3[5]
			ws.append([g4])
			wb.save('train_test2_all_real.xlsx')
	else:
		break
	
