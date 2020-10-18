import pymysql
import pandas as pd


# DB 연결
db = pymysql.connect(
    host = '18.191.246.106', 
    port =3306, 
    user = 'root', 
    password = 'hanium', 
    db='NEWHAN',
    charset='utf8'
    )

cursor = db.cursor(pymysql.cursors.DictCursor)

# INSERT
sql1 = '''INSERT INTO `User_id` (User_id, User_name, Phone_number, Bank, Account, Option1_1, Option1_2)
    VALUES ('1', 'youngchan', '01056891218', 'nonghyup', '01056891218', '1', '1');'''

cursor.execute(sql1)
db.commit()

# SELECT
sql2 = "SELECT * FROM `User_id`;"
cursor.execute(sql2)
result = cursor.fetchall()
result = pd.DataFrame(result)
result


# UPDATE
sql3 = '''UPDATE `User_id`
  SET User_name = 'euihyun' ;'''
cursor.execute(sql3)
db.commit()
