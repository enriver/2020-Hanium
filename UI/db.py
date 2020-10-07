#import pymysql

import requests
from bs4 import BeautifulSoup

codeName='000720'
address='https://finance.naver.com/item/sise.nhn?code=000720'

web=requests.get(address)
soup=BeautifulSoup(web.content,"html.parser")

nowVal=soup.find(id='_nowVal').get_text().strip() #현재가
diff=soup.find('span',class_='tah p11 nv01').get_text().strip() #전일비
rate=soup.find(id='_rate').get_text().strip() #등락률
quant=soup.find(id='_quant').get_text().strip() #거래량
start=soup.find('span',class_='tah p11').get_text().strip() #시가
high=soup.find(id='_high').get_text().strip() #고가
low=soup.find(id='_low').get_text().strip() #저가
gurae=soup.find(id='_amount').get_text().strip() #거래대금

print(nowVal)
print(diff)
print(rate)
print(quant)
print(start)
print(high)
print(low)
print(gurae)


        

'''
class DataBase():

        # DB연결
        conn=pymysql.connect(host='localhost', user='master_ant', password='hanium1234', db='antmaking', charset='utf8')
        cursor= conn.cursor()
           
        sql="INSERT INTO User_id (user_name, account, opt) values (%s, %s, %s)"

        cursor.execute(sql,(user_name,account_num.rstrip(';'),str(optVal)))
        conn.commit()
        conn.close()
        
'''