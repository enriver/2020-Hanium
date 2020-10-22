import pymysql
import pandas as pd

class database():
    
    def __init__(self):
        self.conn = pymysql.connect(
            host = '3.21.41.113', 
            port =3306, 
            user = 'root',
            password = 'hanium', 
            db='NEWHAN', 
            charset='utf8')
        self.cursor = self.conn.cursor()
         
    # user 테이블에 아이디/계좌 넣기
    def user_insert(self,account):
        sql="INSERT INTO USER (user_account) VALUES ("+account+");"
        self.cursor.execute(sql)
        self.conn.commit()


    # user 테이블에 데이터가 있는지 확인    
    def exist_in_user(self,account_num):
        sql = "select EXISTS (select * from USER where user_account="+account_num+");"
        self.cursor.execute(sql)
        self.conn.commit()
        
        db_res = self.cursor.fetchall()
        return db_res[0][0]
    
    # RETAINED_STOCK 테이블에 데이터 넣기
    def retained_insert(self,account, code):    
        sql = "INSERT INTO RETAINED_STOCK (user_account,stock_code) VALUES ("+account+",'"+code+"');"
        self.cursor.execute(sql)
        self.conn.commit()
    
    # RETAINED_STOCK 데이터 삭제
    def retained_delete(self,account):
        sql= "DELETE FROM RETAINED_STOCK WHERE user_account="+account+";"
        self.cursor.execute(sql)
        self.conn.commit()

    # 보유종목 종목명 받아오기 - 파라미터 : (계좌명)
    def retained_get(self,data):
        sql = "SELECT stock_code FROM RETAINED_STOCK WHERE user_account="+data+";"
        self.cursor.execute(sql,data)
        self.conn.commit()
        db_res = self.cursor.fetchall()
        return db_res

    # 관심종목 종목명 받아오기 - 파라미터 : (계좌명)
    def interest_get(self,data):
        sql = "SELECT stock_code FROM INTEREST_STOCK WHERE user_account='"+data+"';"
        self.cursor.execute(sql)
        self.conn.commit()
        db_res = self.cursor.fetchall()
        return db_res

    # 관심종목 삭제 - 파라미터 : (계좌명, 종목코드)
    def interest_delete(self, account_num, interest_stock_code_deleted):
        sql= "DELETE FROM INTEREST_STOCK WHERE user_account='"+account_num+"' and stock_code='"+interest_stock_code_deleted+"';"
        self.cursor.execute(sql)
        self.conn.commit()

        print('관심종목이 삭제되었습니다.')

    # 관심종목 추가 - 파라미터 : (계좌명, 종목코드)
    def interest_insert(self, account_num, interest_stock_code):
        sql = "INSERT INTO INTEREST_STOCK(user_account,stock_code) VALUES ("+account_num+",'"+interest_stock_code+"');"
        self.cursor.execute(sql)
        self.conn.commit()

        print('관심종목이 추가되었습니다.')