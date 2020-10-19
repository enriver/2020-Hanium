# 메인 UI

import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import uic
from Kiwoom import *
import datetime
import requests
from bs4 import BeautifulSoup
import time
import pandas as pd
from bokeh.plotting import figure, save
from bokeh.models.formatters import NumeralTickFormatter
from bokeh.models import HoverTool
from math import pi
import os

#import pymysql

form_class=uic.loadUiType("main.ui")[0]

    
class MainWindow(QMainWindow,form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
        self.kiwoom=Kiwoom()
        self.kiwoom.comm_connect()

        self.timer=QTimer(self)
        self.timer.start(1000)
        self.timer.timeout.connect(self.timeout)

        #로그아웃
        self.logout_btn.clicked.connect(self.logout_clicked)

        #정보 갱신
        self.renew_clicked()
        self.renew_btn.clicked.connect(self.renew_clicked)

        #계좌정보 가져오기
        global account_num
        account_num=self.kiwoom.dynamicCall("GetLoginInfo(QString)",["ACCNO"]).rstrip(';')
        self.cmb_account.addItem(account_num)

        #사용자명 가져오기
        global user_name
        user_name=self.kiwoom.dynamicCall("GetLoginInfo(QString)","USER_NAME")
        self.lbl_user_name.setText(user_name)

        #옵션 선택
        global optVal
        optVal=1
        self.rapidUp_rd.setChecked(True)
        self.rapidUp_rd.clicked.connect(self.groupboxRadFunction)
        self.rapidDown_rd.clicked.connect(self.groupboxRadFunction)
        self.tradeRank_rd.clicked.connect(self.groupboxRadFunction)
        self.tradeASC_rd.clicked.connect(self.groupboxRadFunction)
        self.tradeDESC_rd.clicked.connect(self.groupboxRadFunction)

        #옵션 저장
        self.optSave_btn.clicked.connect(self.optSave_clicked)

        #예수금
        self.kiwoom.set_input_value("계좌번호",account_num)
        self.kiwoom.set_input_value("비밀번호","0000")
        self.kiwoom.comm_rq_data("opw00001_req","opw00001",0,"2000")
        
        self.lbl_deposit.setText(self.kiwoom.d2_deposit)

        #추정자산,총매입금,총평가금,손익,수익률
        self.kiwoom.reset_opw00018_output()
        self.kiwoom.set_input_value("계좌번호",account_num.rstrip(';'))
        self.kiwoom.comm_rq_data("opw00018_req","opw00018",0,"2000")

        self.lbl_purchase.setText(self.kiwoom.opw00018_output['single'][0]) #총매입
        self.lbl_eval_amt.setText(self.kiwoom.opw00018_output['single'][1]) #총평가
        self.lbl_profitLoss.setText(self.kiwoom.opw00018_output['single'][2]) #총손익
        self.lbl_ror.setText(self.kiwoom.opw00018_output['single'][3]) #총수익률
        self.lbl_asset.setText(self.kiwoom.opw00018_output['single'][4]) #총자산

        #종목 자동완성
        code_list=self.kiwoom.dynamicCall("GetCodeListByMarket(QString)",["0"]) #코스피
        code_list2=self.kiwoom.dynamicCall("GetCodeListByMarket(QString)",["10"]) #코스닥

        kospi_code_list=code_list.split(';')
        kosdaq_code_list=code_list2.split(';')
        total_code_list=kospi_code_list+kosdaq_code_list

        total_code_name_list=[]

        global code_dict
        code_dict=dict()

        for x in total_code_list:
            name=self.kiwoom.dynamicCall("GetMasterCodeName(QString)",[x])
            total_code_name_list.append(name)
            code_dict[name]=x

        
        name_completer=QCompleter(total_code_name_list)
        self.lineEdit.setCompleter(name_completer)

        code_completer=QCompleter(total_code_list)
        self.lineEdit_2.setCompleter(code_completer)

        # 종목 검색 받아오기
        self.nameSearch_btn.clicked.connect(self.codeSearch_clicked)
        self.codeSearch_btn.clicked.connect(self.codeSearch_clicked2)

        # 보유자산 종목 관리
        self.check_balance()

        # 관심종목
        self.interest_btn.clicked.connect(self.interest_add_clicked)

        # 관심종목 삭제
        self.del_interest_btn.clicked.connect(self.interest_delete_clicked)

    # 관심종목 추가
    def interest_add_clicked(self):
        global code_dict

        interest_code=self.ItemSearchBox.title()
        print(interest_code)
        
        exist_interest=self.interest_table.findItems(interest_code, Qt.MatchContains)

        if interest_code=='종목명' or interest_code not in code_dict:
            reply=QMessageBox.information(self,"알림","올바른 종목명이 아닙니다",QMessageBox.Yes)

        elif len(exist_interest)>0:
            reply=QMessageBox.information(self,"알림","이미 관심종목으로 추가된 종목입니다.",QMessageBox.Yes)
        else:
            num=self.interest_table.rowCount() ## 후에 DB에서 가져올 예정
            self.interest_table.setRowCount(num+1)
            item=QTableWidgetItem(interest_code)
            self.interest_table.setItem(num,0,item)
            self.interest_table.resizeRowsToContents()
            self.interest_table.setEditTriggers(QAbstractItemView.NoEditTriggers)

    # 관심종목 삭제
    def interest_delete_clicked(self):
        select_interest=self.interest_table.currentRow()

        if select_interest==-1:
            reply=QMessageBox.information(self,"알림","삭제할 종목을 선택해주세요",QMessageBox.Yes)
        else:
            self.interest_table.removeRow(select_interest)
            

    #  종목명 검색
    def codeSearch_clicked(self):
        global code_dict
        codeName=self.lineEdit.text().strip()

        if codeName=="" or codeName not in code_dict:
            reply=QMessageBox.information(self,"알림","종목명을 제대로 입력해주세요",QMessageBox.Yes)

        else:
            self.ItemSearchBox.setTitle(codeName)
            address='https://finance.naver.com/item/sise.nhn?code='+code_dict[codeName]

            web=requests.get(address)
            soup=BeautifulSoup(web.content,"html.parser")

            nowVal=soup.find(id='_nowVal').get_text().strip() #현재가
            rate=soup.find(id='_rate').get_text().strip() #등락률
            quant=soup.find(id='_quant').get_text().strip() #거래량
            start=soup.find('span',class_='tah p11').get_text().strip() #시가
            high=soup.find(id='_high').get_text().strip() #고가
            low=soup.find(id='_low').get_text().strip() #저가
            gurae=soup.find(id='_amount').get_text().strip() #거래대금

            diff=soup.find('td', class_='first')
            diff_=diff.find(class_='blind').get_text().strip() #전일

            self.lbl_nowVal.setText(nowVal)
            self.lbl_diff.setText(diff_)
            self.lbl_rate.setText(rate)
            self.lbl_start.setText(start)
            self.lbl_high.setText(high)
            self.lbl_low.setText(low)
            self.lbl_quant.setText(quant)
            self.lbl_gurae.setText(gurae)

            diff_r=int(diff_.replace(",",""))
            nowVal_r=int(nowVal.replace(",",""))

            difference_=str(abs(diff_r-nowVal_r))
            self.lbl_difference.setText(self.kiwoom.change_format(difference_))

            qPixmapVar=QPixmap()

            if rate[0]=='+':
                qPixmapVar.load("up.jpg")
                self.lbl_upDown.setPixmap(qPixmapVar)

                font=QFont()
                font.setFamily("Bahnschrift")
                font.setPointSize(18)

                self.lbl_nowVal.setFont(font)
                self.lbl_nowVal.setStyleSheet("Color:red")
                self.lbl_difference.setStyleSheet("Color:red")
                self.lbl_rate.setStyleSheet("Color:red")
            elif rate[0]=='-':
                qPixmapVar.load("down.jpg")
                self.lbl_upDown.setPixmap(qPixmapVar)

                font=QFont()
                font.setFamily("Bahnschrift")
                font.setPointSize(18)
                self.lbl_nowVal.setFont(font)

                self.lbl_nowVal.setStyleSheet("Color:blue")
                self.lbl_difference.setStyleSheet("Color:blue")
                self.lbl_rate.setStyleSheet("Color:blue")
            else:
                qPixmapVar.load("none.jpg")
                self.lbl_upDown.setPixmap(qPixmapVar)

                font=QFont()
                font.setFamily("Bahnscrhift")
                font.setPointSize(18)
                self.lbl_nowVal.setFont(font)

                self.lbl_nowVal.setStyleSheet("Color:Black")
        
            now=datetime.datetime.now()
            nowDate=now.strftime("%Y%m%d")

            #데이터프레임
            stock=self.get_ohlcv(code_dict[codeName],nowDate)
            #print(stock)

            inc=stock.close >= stock.open
            dec=stock.open > stock.close
            w=12*60*60*1000

            stock['date']=pd.to_datetime(stock['date'])
            
            #캔들차트
            candle=figure(plot_width=700, plot_height=225, x_axis_type="datetime", tools=['pan, xwheel_zoom','box_zoom','reset','hover'])
            candle.xaxis.major_label_orientation=pi/4
            candle.yaxis.formatter = NumeralTickFormatter(format='0,0')
            candle.grid.grid_line_alpha=0.3

            candle.segment(stock.date, stock.high, stock.date, stock.low, color="black")
            candle.vbar(stock.date[inc],w, stock.open[inc], stock.close[inc], fill_color="red", line_color="red")
            candle.vbar(stock.date[dec],w, stock.open[dec], stock.close[dec], fill_color="blue", line_color="blue")
            
            hover=candle.select(dict(type=HoverTool))
            hover.tooltips=[("Price","$y{0,0}")]  ## columnSource 뭐시기를 쓰면 hover가 가능하지만 하고싶지않은걸
            hover.mode='mouse'
            save(candle, filename="candle.html")

            url=os.getcwd()
            url_changed=url.replace('\\','/')

            self.webEngineView.load(QUrl(url_changed+"/candle.html"))


    # 종목 코드 검색
    def codeSearch_clicked2(self):
        global code_dict
        code=self.lineEdit_2.text().strip()

        c_name=''
        for key, values in code_dict.items():
            if values==code:
                c_name=key
                break

        if code=='' or c_name=='':
            reply=QMessageBox.information(self,"알림","종목코드를 제대로 입력해주세요",QMessageBox.Yes)
        else:
            self.ItemSearchBox.setTitle(c_name)
            address="https://finance.naver.com/item/sise.nhn?code="+code

            web=requests.get(address)
            soup=BeautifulSoup(web.content,"html.parser")

            nowVal=soup.find(id='_nowVal').get_text().strip() #현재가
            rate=soup.find(id='_rate').get_text().strip() #등락률
            quant=soup.find(id='_quant').get_text().strip() #거래량
            start=soup.find('span',class_='tah p11').get_text().strip() #시가
            high=soup.find(id='_high').get_text().strip() #고가
            low=soup.find(id='_low').get_text().strip() #저가
            gurae=soup.find(id='_amount').get_text().strip() #거래대금

            diff=soup.find('td', class_='first')
            diff_=diff.find(class_='blind').get_text().strip() #전일

            diff_r=int(diff_.replace(",",""))
            nowVal_r=int(nowVal.replace(",",""))

            difference_=str(abs(diff_r-nowVal_r))
            self.lbl_difference.setText(self.kiwoom.change_format(difference_))

            self.lbl_nowVal.setText(nowVal)
            self.lbl_rate.setText(rate)
            self.lbl_diff.setText(diff_)
            self.lbl_start.setText(start)
            self.lbl_high.setText(high)
            self.lbl_low.setText(low)
            self.lbl_quant.setText(quant)
            self.lbl_gurae.setText(gurae)

            qPixmapVar=QPixmap()

            if rate[0]=='+':
                qPixmapVar.load("up.jpg")
                self.lbl_upDown.setPixmap(qPixmapVar)

                font=QFont()
                font.setFamily("Bahnschrift")
                font.setPointSize(18)
                self.lbl_nowVal.setFont(font)

                self.lbl_nowVal.setStyleSheet("Color:red")
                self.lbl_difference.setStyleSheet("Color:red")
                self.lbl_rate.setStyleSheet("Color:red")
            elif rate[0]=='-':
                qPixmapVar.load("down.jpg")
                self.lbl_upDown.setPixmap(qPixmapVar)

                font=QFont()
                font.setFamily("Bahnschrift")
                font.setPointSize(18)
                self.lbl_nowVal.setFont(font)

                self.lbl_nowVal.setStyleSheet("Color:blue")
                self.lbl_difference.setStyleSheet("Color:blue")
                self.lbl_rate.setStyleSheet("Color:blue")
            
            else:
                qPixmapVar.load("none.jpg")
                self.lbl_upDown.setPixmap(qPixmapVar)

                font=QFont()
                font.setFamily("Bahnscrhift")
                font.setPointSize(18)
                self.lbl_nowVal.setFont(font)

                self.lbl_nowVal.setStyleSheet("Color:Black")

            now=datetime.datetime.now()
            nowDate=now.strftime("%Y%m%d")

            #데이터프레임
            stock=self.get_ohlcv(code,nowDate)
            #print(stock)

            inc=stock.close >= stock.open
            dec=stock.open > stock.close
            w=12*60*60*1000

            stock['date']=pd.to_datetime(stock['date'])
            
            #캔들차트
            candle=figure(plot_width=700, plot_height=225, x_axis_type="datetime", tools=['pan, xwheel_zoom','box_zoom','reset','hover'])
            candle.xaxis.major_label_orientation=pi/4
            candle.yaxis.formatter = NumeralTickFormatter(format='0,0')
            candle.grid.grid_line_alpha=0.3

            candle.segment(stock.date, stock.high, stock.date, stock.low, color="black")
            candle.vbar(stock.date[inc],w, stock.open[inc], stock.close[inc], fill_color="red", line_color="red")
            candle.vbar(stock.date[dec],w, stock.open[dec], stock.close[dec], fill_color="blue", line_color="blue")
            
            hover=candle.select(dict(type=HoverTool))
            hover.tooltips=[("Price","$y{0,0}")]  ## columnSource 뭐시기를 쓰면 hover가 가능하지만 하고싶지않은걸
            hover.mode='mouse'
            save(candle, filename="candle.html")

            url=os.getcwd()
            url_changed=url.replace('\\','/')

            self.webEngineView.load(QUrl(url_changed+"/candle.html"))


    def get_ohlcv(self, code,start):
        self.kiwoom.ohlcv={'date':[],'open':[],'high':[],'low':[],'close':[],'volume':[]}

        self.kiwoom.set_input_value("종목코드",code)
        self.kiwoom.set_input_value("기준일자",start)
        self.kiwoom.set_input_value("수정주가구분",1)
        self.kiwoom.comm_rq_data("opt10081_req","opt10081",0,"0101")

        while self.kiwoom.remained_data==True:
            time.sleep(0.2)
            self.kiwoom.set_input_value("종목코드",code)
            self.kiwoom.set_input_value("기준일자",start)
            self.kiwoom.set_input_value("수정주가구분",1)
            self.kiwoom.comm_rq_data("opt10081_req","opt10081",2,"0101")

        df=pd.DataFrame(self.kiwoom.ohlcv, columns=['date','open','high','low','close','volume'])
        #, index=self.kiwoom.ohlcv['date']
        df_sorted=df.sort_values(by='date')
        return df_sorted

    def check_balance(self):
        # 총 자산관리
        item=QTableWidgetItem(self.kiwoom.d2_deposit)
        item.setTextAlignment(Qt.AlignVCenter | Qt.AlignRight)
        self.total_table.setItem(0,0,item)
        for i in range(5):
            item=QTableWidgetItem(self.kiwoom.opw00018_output['single'][i])
            item.setTextAlignment(Qt.AlignVCenter | Qt.AlignRight)
            self.total_table.setItem(0,i+1,item)

        self.total_table.resizeRowsToContents()
        self.total_table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        # 보유 종목 관리
        item_count=len(self.kiwoom.opw00018_output['multi'])
        self.have_table.setRowCount(item_count)

        for j in range(item_count):
            row=self.kiwoom.opw00018_output['multi'][j]
            for i in range(len(row)):
                item=QTableWidgetItem(row[i])
                item.setTextAlignment(Qt.AlignVCenter | Qt.AlignRight)
                self.have_table.setItem(j,i,item)
        self.have_table.resizeRowsToContents()
        self.have_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    
    def optSave_clicked(self):
        global optVal

        if optVal==1:
            print("급등")
        elif optVal==2:
            print("급락")
        elif optVal==3:
            print("거래상위")
        elif optVal==4:
            print("거래증가")
        elif optVal==5:
            print("거래감소")

    
    def groupboxRadFunction(self):
        '''
        옵션값  [급등:1 ,급락:2 ,거래상위:3 ,거래증가:4 ,거래감소:5]
        '''
        global optVal
        
        if self.rapidUp_rd.isChecked():
            optVal=1
        elif self.rapidDown_rd.isChecked():
            optVal=2
        elif self.tradeRank_rd.isChecked():
            optVal=3
        elif self.tradeASC_rd.isChecked():
            optVal=4
        elif self.tradeDESC_rd.isChecked():
            optVal=5

    def renew_clicked(self):
        nowTime=datetime.datetime.now().strftime("%H:%M %p")
        self.lbl_criteria.setText(nowTime)
        print(nowTime+" 으로 갱신")

    def logout_clicked(self,event):
        reply=QMessageBox.question(self,"로그아웃","로그아웃 하시겠습니까?",QMessageBox.Yes|QMessageBox.No,QMessageBox.Yes)

        if reply ==QMessageBox.Yes:
            self.close()
        else:
            pass

    def timeout(self):
        current_time=QTime.currentTime()
        now=datetime.datetime.now()
        text_today=now.strftime('%Y-%m-%d')
        text_day=datetime.datetime.today().weekday()

        t=['월','화','수','목','금','토','일']

        text_time=current_time.toString("hh:mm:ss")
        time_msg="현재시간: "+text_time

        state=self.kiwoom.get_connect_state()

        if state==1:
            state_msg="서버 연결 중"
        else:
            state_msg="서버 미 연결 중"

        self.statusbar.showMessage(state_msg+" | "+time_msg)
        self.lbl_date_time.setText(text_today+" ("+t[text_day]+")  "+text_time)
        

if __name__ == '__main__': 
    app=QApplication(sys.argv)
    mainWindow=MainWindow()
    mainWindow.show()
    app.exec_()