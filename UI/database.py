import pymysql
import pandas as pd

class database():
    
    def __init__(self):
        self.conn =pymysql.connect(host = '18.225.8.69', port =3306, user = 'root', password = 'hanium', db='NEWHAN', charset='utf8')
        self.cursor = self.conn.cursor()

    # 보유종목 종목명 받아오기 - 파라미터 : (계좌명)
    def retained_get(self,data):
        sql = "SELECT stock_code FROM RETAINED_STOCK WHERE user_account=$s;"
        self.cursor.execute(sql,data)
        self.conn.commit()
        db_res = self.cursor.fetchall()
        return db_res

    # 관심종목 종목명 받아오기 - 파라미터 : (계좌명)
    def interest_get(self,data):
        sql = "SELECT stock_code FROM INTEREST_STOCK WHERE user_account=$s;"
        self.cursor.execute(sql,data)
        self.conn.commit()
        db_res = self.cursor.fetchall()
        return db_res

    # 보유종목 삭제 - 파라미터 : (계좌명, 종목코드)
    def ratained_delete(self, data):
        sql= "DELETE FROM RETAINED_STOCK WHERE user_account=$s and stock_code=$s;"
        self.cursor.execute(sql,data)
        self.conn.commit()
        
        print('보유종목이 삭제되었습니다.')

    # 관심종목 삭제 - 파라미터 : (계좌명, 종목코드)
    def interest_delete(self, data):
        sql= "DELETE FROM INTEREST_STOCK WHERE user_account=$s and stock_code=$s;"
        self.cursor.execute(sql,data)
        self.conn.commit()

        print('관심종목이 삭제되었습니다.')

    # 관심종목 추가 - 파라미터 : (계좌명, 종목코드)
    def interest_insert(self, data):
        sql = "INSERT INTO INTEREST_STOCK(user_account,stock_code) VALUES ($s,$s);"
        self.cursor.execute(sql,data)
        self.conn.commit()

        print('관심종목이 추가되었습니다.')