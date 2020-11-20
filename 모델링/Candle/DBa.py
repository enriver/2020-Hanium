import pymysql
import numpy as np
import os
import time
import pandas as pd

class database():
    
    def __init__(self):
        self.conn =pymysql.connect(host = 'localhost', port =3306, user = 'root', password = 'hanium', db='NEWHAN', charset='utf8')
        self.cursor = self.conn.cursor()


    def insert(self, data):
        sql = """INSERT INTO CRAWL_STOCK(stock_code, naver_option) VALUES (%s,%s);"""
        self.cursor.executemany(sql,data)
        self.conn.commit()
    
    def insert_buylist(self, data): 
        sql = """INSERT INTO BUY_LIST(stock_code, predict_value, predict_cnn, up_down, naver_option) VALUES (%s,%s,%s,%s,%s);"""
        self.cursor.executemany(sql,data)
        self.conn.commit()

    def insert_selllist(self, data): 
        sql = """INSERT INTO SELL_LIST(stock_code, predict_value, predict_cnn, up_down) VALUES (%s,%s,%s,%s);"""
        self.cursor.executemany(sql,data)
        self.conn.commit()

    def check_crawl(self, code):
        sql = "select naver_option from CRAWL_STOCK where stock_code=%s"
        self.cursor.execute(sql, code)
        self.conn.commit()
        res = self.cursor.fetchall()
        return res



    def copy_table(self):
        sql = """INSERT INTO VIEW_INTEREST (select * from INTEREST_STOCK)"""
        self.cursor.execute(sql)
        self.conn.commit()
        sql = """INSERT INTO VIEW_RETAINED (select * from RETAINED_STOCK)"""
        self.cursor.execute(sql)
        self.conn.commit()

        

    def reset(self,db):
        sql = "TRUNCATE "+db+";"
        self.cursor.execute(sql)
        self.conn.commit()

    def crawl_get(self):
        sql = "SELECT * FROM CRAWL_STOCK;"
        self.cursor.execute(sql)
        self.conn.commit()
        db_res = self.cursor.fetchall()
        return db_res
        
    def retained_get(self):
        sql = "SELECT * FROM RETAINED_STOCK;"
        self.cursor.execute(sql)
        self.conn.commit()
        db_res = self.cursor.fetchall()
        return db_res

    def interest_get(self):
        sql = "SELECT * FROM INTEREST_STOCK;"
        self.cursor.execute(sql)
        self.conn.commit()
        db_res = self.cursor.fetchall()
        return db_res



        
        
    
