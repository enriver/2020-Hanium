# 메인 UI

import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import uic
from Kiwoom import *
from database import *
import datetime
import requests
from bs4 import BeautifulSoup
import time
import pandas as pd
from bokeh.plotting import figure, save
from bokeh.models.formatters import NumeralTickFormatter
from bokeh.models import HoverTool, ColumnDataSource
from math import pi
import os
import numpy as np

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
        nowTime=datetime.datetime.now().strftime("%H:%M %p")
        self.lbl_criteria.setText(nowTime)
        self.renew_btn.clicked.connect(self.renew_clicked)

        #계좌정보 가져오기
        global account_num
        account_num=self.kiwoom.dynamicCall("GetLoginInfo(QString)",["ACCNO"]).rstrip(';')
        self.cmb_account.addItem(account_num)

        #사용자명 가져오기
        global user_name
        user_name=self.kiwoom.dynamicCall("GetLoginInfo(QString)","USER_NAME")
        self.lbl_user_name.setText(user_name)

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

        # DB 연결
        self.db=database()

        '''
        ###### 사용자 등록 여부 확인 #####
        '''
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

        # 보유자산 종목 관리
        self.check_balance()

        #옵션 선택
        global optVal
        optVal=0

        self.rapidUp_rd.setChecked(True)
        self.rapidUp_rd.clicked.connect(self.groupboxRadFunction)
        self.rapidDown_rd.clicked.connect(self.groupboxRadFunction)
        self.tradeRank_rd.clicked.connect(self.groupboxRadFunction)
        self.tradeASC_rd.clicked.connect(self.groupboxRadFunction)
        self.tradeDESC_rd.clicked.connect(self.groupboxRadFunction)

        # 첫 로그인 체크
        user_check=self.db.exist_in_user(account_num)

        if user_check==0: # 첫 로그인일때
            print('첫 로그인')
            self.db.user_insert(account_num)
                    
        else: # 첫 로그인이 아닐때
            print('첫 로그인이 아닙니다.')

            # SELL_LIST에서 보유종목 받아오기 - 주식코드, 예측값, 업다운
            global sell_retain
            sell_retain=self.db.get_sell_list(account_num)
            # BUY_LIST에서 보유종목 받아오기 - 주식코드, 예측값, 업다운
            global buy_retain
            buy_retain=self.db.get_buy_list_retained(account_num)
            # BUY_LIST에서 관심종목 받아오기 - 주식코드, 예측값, 업다운
            global buy_interest
            buy_interest=self.db.get_buy_list_interest(account_num)
            
            global len_sell_retain
            global len_buy_retain
            global len_buy_interest

            len_sell_retain=len(sell_retain)
            len_buy_retain=len(buy_retain)
            len_buy_interest=len(buy_interest)

            global len_total
            len_total=len_sell_retain+len_buy_retain+len_buy_interest
            self.main_table.setRowCount(len_total)

            # 메인화면에 보유종목(sell) 띄우기
            if len_sell_retain > 0:
    
                for i in range(len_sell_retain):
                    result=self.get_mainView(sell_retain[i][0]) # 메인뷰에 띄울것들 호출
                    
                    for j in range(len(result.columns)):
                        item=QTableWidgetItem(result[result.columns[j]][0])
                        item.setTextAlignment(Qt.AlignCenter)
                        self.main_table.setItem(i,j,item)
                    
                    item=QTableWidgetItem('매도')
                    item.setTextAlignment(Qt.AlignCenter)
                    self.main_table.setItem(i,8,item)
                    item=QTableWidgetItem(str(sell_retain[i][1]))
                    item.setTextAlignment(Qt.AlignCenter)
                    self.main_table.setItem(i,9,item)

                    if sell_retain[i][2] == 2 :
                        item=QTableWidgetItem('상')
                    elif sell_retain[i][2] == 1 :
                        item=QTableWidgetItem('중')
                    else:
                        item=QTableWidgetItem('하')
                    item.setTextAlignment(Qt.AlignCenter)
                    self.main_table.setItem(i,10,item)

                    item=QTableWidgetItem('보유')
                    item.setTextAlignment(Qt.AlignCenter)
                    self.main_table.setItem(i,11,item)

            # 메인화면에 보유종목(buy) 띄우기
            if len_buy_retain > 0:
        
                for i in range(len_buy_retain):
                    result=self.get_mainView(buy_retain[i][0]) # 메인뷰에 띄울것들 호출
                    
                    for j in range(len(result.columns)):
                        item=QTableWidgetItem(result[result.columns[j]][0])
                        item.setTextAlignment(Qt.AlignCenter)
                        self.main_table.setItem(i+len_sell_retain,j,item)
                    
                    item=QTableWidgetItem('매수')
                    item.setTextAlignment(Qt.AlignCenter)
                    self.main_table.setItem(i+len_sell_retain,8,item)
                    item=QTableWidgetItem(str(buy_retain[i][1]))
                    item.setTextAlignment(Qt.AlignCenter)
                    self.main_table.setItem(i+len_sell_retain,9,item)

                    if buy_retain[i][2] == 2 :
                        item=QTableWidgetItem('상')
                    elif buy_retain[i][2] == 1 :
                        item=QTableWidgetItem('중')
                    else:
                        item=QTableWidgetItem('하')
                    item.setTextAlignment(Qt.AlignCenter)
                    self.main_table.setItem(i+len_sell_retain,10,item)

                    item=QTableWidgetItem('보유')
                    item.setTextAlignment(Qt.AlignCenter)
                    self.main_table.setItem(i+len_sell_retain,11,item)

            # 메인화면에 관심종목(buy) 띄우기
            if len_buy_interest > 0:
        
                for i in range(len_buy_interest):
                    result=self.get_mainView(buy_interest[i][0]) # 메인뷰에 띄울것들 호출
                    
                    for j in range(len(result.columns)):
                        item=QTableWidgetItem(result[result.columns[j]][0])
                        item.setTextAlignment(Qt.AlignCenter)
                        self.main_table.setItem(i+len_sell_retain+len_buy_retain,j,item)
                    
                    item=QTableWidgetItem('매수')
                    item.setTextAlignment(Qt.AlignCenter)
                    self.main_table.setItem(i+len_sell_retain+len_buy_retain,8,item)
                    item=QTableWidgetItem(str(buy_interest[i][1]))
                    item.setTextAlignment(Qt.AlignCenter)
                    self.main_table.setItem(i+len_sell_retain+len_buy_retain,9,item)

                    if buy_interest[i][2] == 2 :
                        item=QTableWidgetItem('상')
                    elif buy_interest[i][2] == 1 :
                        item=QTableWidgetItem('중')
                    else:
                        item=QTableWidgetItem('하')
                    item.setTextAlignment(Qt.AlignCenter)
                    self.main_table.setItem(i+len_sell_retain+len_buy_retain,10,item)

                    item=QTableWidgetItem('관심')
                    item.setTextAlignment(Qt.AlignCenter)
                    self.main_table.setItem(i+len_sell_retain+len_buy_retain,11,item)

            
            up_list=self.db.get_buy_list_crawl(0)

            # 초기 메인화면에 급등 띄우기
            if len(up_list) > 0:
        
                for i in range(len(up_list)):
                    self.main_table.insertRow(i+len_total)
                    result=self.get_mainView(up_list[i][0]) # 메인뷰에 띄울것들 호출
                    
                    for j in range(len(result.columns)):
                        item=QTableWidgetItem(result[result.columns[j]][0])
                        item.setTextAlignment(Qt.AlignCenter)
                        self.main_table.setItem(i+len_total,j,item)
                    
                    item=QTableWidgetItem('매수')
                    item.setTextAlignment(Qt.AlignCenter)
                    self.main_table.setItem(i+len_total,8,item)
                    item=QTableWidgetItem(str(up_list[i][1]))
                    item.setTextAlignment(Qt.AlignCenter)
                    self.main_table.setItem(i+len_total,9,item)

                    if up_list[i][2] == 2 :
                        item=QTableWidgetItem('상')
                    elif up_list[i][2] == 1 :
                        item=QTableWidgetItem('중')
                    else:
                        item=QTableWidgetItem('하')
                    item.setTextAlignment(Qt.AlignCenter)
                    self.main_table.setItem(i+len_total,10,item)

                    item=QTableWidgetItem('급등')
                    item.setTextAlignment(Qt.AlignCenter)
                    self.main_table.setItem(i+len_total,11,item)
                
            self.main_table.resizeRowsToContents()
            self.main_table.setEditTriggers(QAbstractItemView.NoEditTriggers)         

            # 관심종목 테이블 추가
            interest_list=self.db.interest_get(account_num)
            if len(interest_list) > 0:
                self.interest_table.setRowCount(len(interest_list))

                for i in range(len(interest_list)):
                    for key,values in code_dict.items():
                        if values==interest_list[i][0]:
                            stock_name=key
                            break

                    item=QTableWidgetItem(stock_name)
                    self.interest_table.setItem(i,0,item)
                    self.interest_table.resizeRowsToContents()
                    self.interest_table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        #Dialog 불러오기
        self.dialog=uic.loadUi("trade.ui")
        self.dialog.cmb_account.addItem(account_num) #계좌번호 받아오기
        self.dialog.cmb_order.currentIndexChanged.connect(self.order_set)
        self.dialog.cmb_kinds.currentIndexChanged.connect(self.kinds_set)

        self.dialog.btn_order.clicked.connect(self.send_order)

        #옵션 선택
        self.optSave_btn.clicked.connect(self.optSave_clicked)

        # 종목 검색 받아오기
        self.nameSearch_btn.clicked.connect(self.codeSearch_clicked)
        self.codeSearch_btn.clicked.connect(self.codeSearch_clicked2)

        # 관심종목 추가
        self.interest_btn.clicked.connect(self.interest_add_clicked)

        # 관심종목 삭제
        self.del_interest_btn.clicked.connect(self.interest_delete_clicked)

        # 매매
        self.buySell_btn_1.clicked.connect(self.buySellClicked1)
        self.buySell_btn_2.clicked.connect(self.buySellClicked2)
        self.buySell_btn_3.clicked.connect(self.buySellClicked3)

        # 보유종목 갱신
        self.btn_retained_renew.clicked.connect(self.update_retained_tell)

    # 메인화면 매매
    def buySellClicked1(self):
        select_code=self.main_table.currentRow()

        if select_code==-1:
            reply=QMessageBox.information(self,"알림","매매할 종목을 선택해주세요",QMessageBox.Yes) 
        else:
            self.update_retained()

            select_item=self.main_table.item(select_code,0).text()
            self.dialog.label_code.setText(code_dict[select_item]) #종목코드 받아오기 
            self.dialog.label_codeName.setText(select_item) #종목명 받아오기 
            self.dialog.show()
        

    # 검색된 종목 매매
    def buySellClicked2(self):
        global code_dict

        select_code=self.ItemSearchBox.title()


        if select_code == '종목명' or select_code not in code_dict:
            reply=QMessageBox.information(self,"알림","종목 검색 후 매매를 시도하세요",QMessageBox.Yes)
        else:
            self.update_retained()
            self.dialog.label_code.setText(code_dict[select_code]) #종목코드 받아오기 
            self.dialog.label_codeName.setText(select_code) #종목명 받아오기 
            self.dialog.show()
            
    # 보유종목 매매
    def buySellClicked3(self):
        select_code=self.have_table.currentRow()

        if select_code==-1:
            reply=QMessageBox.information(self,"알림","매매할 종목을 선택해주세요",QMessageBox.Yes)
        else:
            self.update_retained()

            select_item=self.have_table.item(select_code,0).text()
            self.dialog.label_code.setText(code_dict[select_item]) #종목코드 받아오기 
            self.dialog.label_codeName.setText(select_item) #종목명 받아오기 
            self.dialog.show()

    # 매매함수
    def send_order(self):
        order_type_lookup = {'신규매수': 1, '신규매도': 2, '매수취소': 3, '매도취소': 4}
        hoga_lookup = {'지정가': "00", '시장가': "03"}

        account = self.dialog.cmb_account.currentText()
        order_type = self.dialog.cmb_order.currentText()
        code = self.dialog.label_code.text()
        hoga = self.dialog.cmb_kinds.currentText()
        num = self.dialog.spin_num.value()
        price = self.dialog.spin_price.value()

        order_no=''
        # 시장가를 선택했을 경우 현재가를 반환해줄것 __init__ self로 구현하면 편할듯

        if order_type=='신규매수':
            if num<1:
                reply=QMessageBox.information(self,"알림","1주 이상 신청할 수 있습니다.",QMessageBox.Yes) 
            else:
                self.kiwoom.send_order("send_order_req", "0101", account, order_type_lookup[order_type], code, num, price, hoga_lookup[hoga], order_no)
                reply=QMessageBox.information(self,"알림","<"+self.dialog.label_codeName.text()+"> 를 "+str(num)+"주 신규매수 신청하였습니다.",QMessageBox.Yes) 

        elif order_type=='신규매도':
            if num<1:
                reply=QMessageBox.information(self,"알림","1주 이상 신청할 수 있습니다.",QMessageBox.Yes) 
            elif sell_num<num:
                reply=QMessageBox.information(self,"알림","보유주 이상 매도할 수 없습니다.",QMessageBox.Yes) 
            else:
                self.kiwoom.send_order("send_order_req", "0101", account, order_type_lookup[order_type], code, num, price, hoga_lookup[hoga], order_no)
                reply=QMessageBox.information(self,"알림","<"+self.dialog.label_codeName.text()+"> 를 "+str(num)+"주 신규매도 신청하였습니다.",QMessageBox.Yes) 
        elif order_type=='매수취소':
            if num<1:
                reply=QMessageBox.information(self,"알림","1주 이상 신청할 수 있습니다.",QMessageBox.Yes) 
            else:
                self.kiwoom.send_order("send_order_req", "0101", account, order_type_lookup[order_type], code, num, price, hoga_lookup[hoga], order_no)
                reply=QMessageBox.information(self,"알림","<"+self.dialog.label_codeName.text()+"> 를 "+str(num)+"주 매수취소 신청하였습니다.",QMessageBox.Yes) 
        else:
            if num<1:
                reply=QMessageBox.information(self,"알림","1주 이상 신청할 수 있습니다.",QMessageBox.Yes) 
            else:
                self.kiwoom.send_order("send_order_req", "0101", account, order_type_lookup[order_type], code, num, price, hoga_lookup[hoga], order_no)
                reply=QMessageBox.information(self,"알림","<"+self.dialog.label_codeName.text()+"> 를"+str(num)+"주 매도취소 신청하였습니다.",QMessageBox.Yes) 

        self.update_retained()

    # 보유종목 업데이트
    def update_retained(self):
        self.db.retained_delete(account_num)

        self.kiwoom.reset_opw00018_output()
        self.kiwoom.set_input_value("계좌번호",account_num.rstrip(';'))
        self.kiwoom.comm_rq_data("opw00018_req","opw00018",0,"2000")
        
        item_count=len(self.kiwoom.opw00018_output['multi'])
        self.have_table.setRowCount(item_count)

        for j in range(item_count):
            row=self.kiwoom.opw00018_output['multi'][j]
            row2=self.kiwoom.opw00018_output['retained'][j]
            self.db.retained_insert(account_num,row2[0][1:])

            for i in range(len(row)):
                item=QTableWidgetItem(row[i])
                item.setTextAlignment(Qt.AlignVCenter|Qt.AlignRight)
                self.have_table.setItem(j,i,item)
        self.have_table.resizeRowsToContents()
        self.have_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        
        print('보유종목 delete / update')

    # 보유종목 갱신 알림
    def update_retained_tell(self):
        self.db.retained_delete(account_num)

        self.kiwoom.reset_opw00018_output()
        self.kiwoom.set_input_value("계좌번호",account_num.rstrip(';'))
        self.kiwoom.comm_rq_data("opw00018_req","opw00018",0,"2000")
        
        item_count=len(self.kiwoom.opw00018_output['multi'])
        self.have_table.setRowCount(item_count)

        for j in range(item_count):
            row=self.kiwoom.opw00018_output['multi'][j]
            row2=self.kiwoom.opw00018_output['retained'][j]
            self.db.retained_insert(account_num,row2[0][1:])

            for i in range(len(row)):
                item=QTableWidgetItem(row[i])
                item.setTextAlignment(Qt.AlignVCenter|Qt.AlignRight)
                self.have_table.setItem(j,i,item)
        self.have_table.resizeRowsToContents()
        self.have_table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        reply=QMessageBox.information(self,"알림","보유종목이 갱신되었습니다.",QMessageBox.Yes)
        
        print('보유종목 갱신')

    # 관심종목 추가
    def interest_add_clicked(self):
        global code_dict
        global account_num

        interest_name=self.ItemSearchBox.title()
        print(interest_name)
        
        exist_interest=self.interest_table.findItems(interest_name, Qt.MatchContains)

        retained_db=list()
        retained_list=self.db.retained_get(account_num)
        
        # 보유 종목이 있을 때
        if len(retained_list) > 0:
            for i in range(len(retained_list)):
                retained_db.append(retained_list[i][0])

            if interest_name=='종목명' or interest_name not in code_dict:
                reply=QMessageBox.information(self,"알림","올바른 종목명이 아닙니다",QMessageBox.Yes)

            elif len(exist_interest)>0:
                reply=QMessageBox.information(self,"알림","이미 관심종목으로 추가된 종목입니다.",QMessageBox.Yes)

            elif code_dict[interest_name] in retained_db:
                reply=QMessageBox.information(self,"알림","이미 보유하신 종목입니다.",QMessageBox.Yes)
            else:
                num=self.interest_table.rowCount() ## 후에 DB에서 가져올 예정
                self.interest_table.setRowCount(num+1)
                item=QTableWidgetItem(interest_name)
                self.interest_table.setItem(num,0,item)
                self.db.interest_insert(account_num,code_dict[interest_name])
                self.interest_table.resizeRowsToContents()
                self.interest_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
                reply=QMessageBox.information(self,"알림","<"+interest_name+">  관심종목으로 추가되었습니다.",QMessageBox.Yes)

        # 보유 종목이 없을 때
        else:
            if interest_name=='종목명' or interest_name not in code_dict:
                reply=QMessageBox.information(self,"알림","올바른 종목명이 아닙니다",QMessageBox.Yes)

            elif len(exist_interest)>0:
                reply=QMessageBox.information(self,"알림","이미 관심종목으로 추가된 종목입니다.",QMessageBox.Yes)

            else:
                num=self.interest_table.rowCount() ## 후에 DB에서 가져올 예정
                self.interest_table.setRowCount(num+1)
                item=QTableWidgetItem(interest_name)
                self.interest_table.setItem(num,0,item)
                self.db.interest_insert(account_num,code_dict[interest_name])
                self.interest_table.resizeRowsToContents()
                self.interest_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
                reply=QMessageBox.information(self,"알림","<"+interest_name+">  관심종목으로 추가되었습니다.",QMessageBox.Yes)

    # 관심종목 삭제
    def interest_delete_clicked(self):
        select_interest=self.interest_table.currentRow()

        if select_interest==-1:
            reply=QMessageBox.information(self,"알림","삭제할 종목을 선택해주세요",QMessageBox.Yes)
        else:
            select_item=self.interest_table.currentItem().text()

            self.interest_table.removeRow(select_interest)
            self.db.interest_delete(account_num,code_dict[select_item])
            reply=QMessageBox.information(self,"알림","<"+select_item+">  관심종목에서 삭제되었습니다.",QMessageBox.Yes)
            

    #  종목명 검색
    def codeSearch_clicked(self):
        global code_dict
        codeName=self.lineEdit.text().strip()

        if codeName=="" or codeName not in code_dict:
            reply=QMessageBox.information(self,"알림","종목명을 제대로 입력해주세요",QMessageBox.Yes)

        else:
            self.ItemSearchBox.setTitle(codeName)
            self.crawlAndChart(code_dict[codeName])


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
            self.crawlAndChart(code)
    
    def crawlAndChart(self,stock_code):
            result=self.get_searchView(stock_code)
            
            
            self.lbl_nowVal.setText(self.kiwoom.change_format(str(result.nowVal[0])))
            self.lbl_rate.setText(result.rate[0])
            self.lbl_start.setText(result.open[0])
            self.lbl_high.setText(result.high[0])
            self.lbl_low.setText(result.low[0])
            self.lbl_quant.setText(result.quant[0])
            self.lbl_up.setText(result.up[0])
            self.lbl_down.setText(result.down[0])
            self.lbl_per.setText(result.per[0])
            self.lbl_roe.setText(result.roe[0])

            self.lbl_up.setStyleSheet("Color:red")
            self.lbl_down.setStyleSheet("Color:blue")

            qPixmapVar=QPixmap()
            
            if result.rate[0][0]=='+':
                qPixmapVar.load("up.jpg")
                self.lbl_upDown.setPixmap(qPixmapVar)

                font=QFont()
                font.setFamily("Bahnschrift")
                font.setPointSize(18)
                
                self.lbl_difference.setText(self.kiwoom.change_format(str(result['diff'][0])))
                self.lbl_diff.setText(self.kiwoom.change_format(str(result.nowVal[0]-result['diff'][0])))

                self.lbl_nowVal.setFont(font)
                self.lbl_nowVal.setStyleSheet("Color:red")
                self.lbl_difference.setStyleSheet("Color:red")
                self.lbl_rate.setStyleSheet("Color:red")

            elif result.rate[0][0]=='-':
                qPixmapVar.load("down.jpg")
                self.lbl_upDown.setPixmap(qPixmapVar)

                font=QFont()
                font.setFamily("Bahnschrift")
                font.setPointSize(18)
                
                self.lbl_difference.setText(self.kiwoom.change_format(str(result['diff'][0])))
                self.lbl_diff.setText(self.kiwoom.change_format(str(result.nowVal[0]+result['diff'][0])))

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
                self.lbl_difference.setText('0')
                self.lbl_diff.setText(self.kiwoom.change_format(str(result.nowVal[0])))

            now=datetime.datetime.now()
            nowDate=now.strftime("%Y%m%d")

            #데이터프레임
            stock=self.get_ohlcv(stock_code,nowDate)
            #print(stock)

            w=12*60*60*1000
            
            stock['dates']=pd.to_datetime(stock['date'])
            stock[['date']]=stock[['date']].applymap(str).applymap(lambda x: "{}-{}-{}".format(x[0:4],x[4:6],x[6:]))
            
            inc=stock.close >= stock.open
            dec=stock.open > stock.close
            
            TOOLTIPS=[
                    ('날짜','@date'),
                    ('시가','@open{0,0}'), 
                    ('고가','@high{0,0}'),
                    ('저가','@low{0,0}'),
                    ('종가','@close{0,0}'),
                    ]

            stock['dateinc'] = stock.dates[inc]
            stock['openinc'] = stock.open[inc]
            stock['closeinc'] = stock.close[inc]
            stock['datedec'] = stock.dates[dec]
            stock['opendec'] = stock.open[dec]
            stock['closedec'] = stock.close[dec]

            source=ColumnDataSource(stock)
            # 캔들차트
            candle=figure(plot_width=700, plot_height=225, x_axis_type="datetime", tools=['pan', 'xwheel_zoom','box_zoom','reset','hover','crosshair'])
            candle.xaxis.major_label_orientation=pi/4
            candle.yaxis.formatter = NumeralTickFormatter(format='0,0')
            candle.grid.grid_line_alpha=0.3

            candle.segment('dates','high','dates','low', color="black",source=source)
            candle.vbar('dateinc',w,'openinc','closeinc', fill_color="red", line_color="red",source=source, name='up')
            candle.vbar('datedec',w,'opendec','closedec', fill_color="blue", line_color="blue",source=source, name='down')

            hover=candle.select(dict(type=HoverTool))
            hover.tooltips=TOOLTIPS
            hover.names=['up','down']
        
            hover.mode='vline'
            save(candle, filename="candle.html")

            url=os.getcwd()
            url_changed=url.replace('\\','/')

            self.webEngineView.load(QUrl(url_changed+"/candle.html"))
            
    def get_searchView(self,code):
        self.kiwoom.searchView={'nowVal':[], 'diff':[],'rate':[],'quant':[],'open':[],'high':[],'low':[],'up':[],'down':[],'per':[],'roe':[]} 
        self.kiwoom.set_input_value("종목코드",code)
        self.kiwoom.comm_rq_data("opt10001_req","opt10001",0,"0101")

        df=pd.DataFrame(self.kiwoom.searchView, columns=['nowVal','diff','rate','quant','open','high','low','up','down','per','roe'])

        return df

    def get_mainView(self,code):
        time.sleep(0.2)
        self.kiwoom.mainView={'name':[],'nowVal':[], 'diff':[],'rate':[],'quant':[],'open':[],'high':[],'low':[]}
        self.kiwoom.set_input_value("종목코드",code)
        self.kiwoom.comm_rq_data("mainView","opt10001",0,"0101")

        df=pd.DataFrame(self.kiwoom.mainView, columns=['name','nowVal','diff','rate','quant','open','high','low'])

        return df

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

    # order_type 에 따른 값 변경
    def order_set(self):
        order_type=self.dialog.cmb_order.currentText()

        if order_type=='신규매도':
            self.kiwoom.reset_opw00018_output()
            self.kiwoom.set_input_value("계좌번호",account_num.rstrip(';'))
            self.kiwoom.comm_rq_data("opw00018_req","opw00018",0,"2000")
        
            item_count=len(self.kiwoom.opw00018_output['retained'])

            retain_dict=dict()
            for j in range(item_count):
                row=self.kiwoom.opw00018_output['retained'][j]
                retain_dict[row[1]]=int(row[2])

            if self.dialog.label_codeName.text() in retain_dict:
                global sell_num
                sell_num=retain_dict[self.dialog.label_codeName.text()]
                self.dialog.spin_num.setValue(sell_num)
            else:
                self.dialog.btn_order.setEnabled(False)

        else:
            self.dialog.spin_num.setValue(0)
            self.dialog.btn_order.setEnabled(True)


    # 호가에 따른 값 변경
    def kinds_set(self):
        global nowVal
        hoga=self.dialog.cmb_kinds.currentText()

        if hoga=='시장가':
            self.dialog.spin_price.setValue(0)
            self.dialog.spin_price.setEnabled(False)
        else:
            self.dialog.spin_price.setEnabled(True)

    # 총 자산관리
    def check_balance(self):
        item=QTableWidgetItem(self.kiwoom.d2_deposit)
        item.setTextAlignment(Qt.AlignVCenter|Qt.AlignRight)
        self.total_table.setItem(0,0,item)
        for i in range(5):
            item=QTableWidgetItem(self.kiwoom.opw00018_output['single'][i])
            item.setTextAlignment(Qt.AlignVCenter|Qt.AlignRight)
            self.total_table.setItem(0,i+1,item)

        self.total_table.resizeRowsToContents()
        self.total_table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        # 보유 종목 관리
        self.db.retained_delete(account_num)

        item_count=len(self.kiwoom.opw00018_output['multi'])
        self.have_table.setRowCount(item_count)

        for j in range(item_count):
            row=self.kiwoom.opw00018_output['multi'][j]
            row2=self.kiwoom.opw00018_output['retained'][j]
            self.db.retained_insert(account_num,row2[0][1:])

            for i in range(len(row)):
                item=QTableWidgetItem(row[i])
                if i==0:
                    item.setTextAlignment(Qt.AlignCenter)
                else:
                    item.setTextAlignment(Qt.AlignVCenter|Qt.AlignRight)
                self.have_table.setItem(j,i,item)
        self.have_table.resizeRowsToContents()
        self.have_table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        
    
    # 옵션변경에 따른 함수
    def optSave_clicked(self):
        global optVal
        self.main_table.setRowCount(len_total)

        if optVal==0:
            global up_list
            up_list=self.db.get_buy_list_crawl(0)

            # 메인화면에 급등 띄우기
            if len(up_list) > 0:
        
                for i in range(len(up_list)):
                    self.main_table.insertRow(i+len_total)
                    result=self.get_mainView(up_list[i][0]) # 메인뷰에 띄울것들 호출
                    
                    for j in range(len(result.columns)):
                        item=QTableWidgetItem(result[result.columns[j]][0])
                        item.setTextAlignment(Qt.AlignCenter)
                        self.main_table.setItem(i+len_total,j,item)
                    
                    item=QTableWidgetItem('매수')
                    item.setTextAlignment(Qt.AlignCenter)
                    self.main_table.setItem(i+len_total,8,item)
                    item=QTableWidgetItem(str(up_list[i][1]))
                    item.setTextAlignment(Qt.AlignCenter)
                    self.main_table.setItem(i+len_total,9,item)

                    if up_list[i][2] == 2 :
                        item=QTableWidgetItem('상')
                    elif up_list[i][2] == 1 :
                        item=QTableWidgetItem('중')
                    else:
                        item=QTableWidgetItem('하')
                    item.setTextAlignment(Qt.AlignCenter)
                    self.main_table.setItem(i+len_total,10,item)

                    item=QTableWidgetItem('급등')
                    item.setTextAlignment(Qt.AlignCenter)
                    self.main_table.setItem(i+len_total,11,item)
                
            self.main_table.resizeRowsToContents()
            self.main_table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        elif optVal==1:
            global down_list
            down_list=self.db.get_buy_list_crawl(1)

            # 메인화면에 급락 띄우기
            if len(down_list) > 0:
        
                for i in range(len(down_list)):
                    self.main_table.insertRow(i+len_total)
                    result=self.get_mainView(down_list[i][0]) # 메인뷰에 띄울것들 호출
                    
                    for j in range(len(result.columns)):
                        item=QTableWidgetItem(result[result.columns[j]][0])
                        item.setTextAlignment(Qt.AlignCenter)
                        self.main_table.setItem(i+len_total,j,item)
                    
                    item=QTableWidgetItem('매수')
                    item.setTextAlignment(Qt.AlignCenter)
                    self.main_table.setItem(i+len_total,8,item)
                    item=QTableWidgetItem(str(down_list[i][1]))
                    item.setTextAlignment(Qt.AlignCenter)
                    self.main_table.setItem(i+len_total,9,item)

                    if down_list[i][2] == 2 :
                        item=QTableWidgetItem('상')
                    elif down_list[i][2] == 1 :
                        item=QTableWidgetItem('중')
                    else:
                        item=QTableWidgetItem('하')
                    item.setTextAlignment(Qt.AlignCenter)
                    self.main_table.setItem(i+len_total,10,item)

                    item=QTableWidgetItem('급락')
                    item.setTextAlignment(Qt.AlignCenter)
                    self.main_table.setItem(i+len_total,11,item)
        
        elif optVal==2:
            global top_list
            top_list=self.db.get_buy_list_crawl(2)

            # 메인화면에 거래상위 띄우기
            if len(top_list) > 0:
        
                for i in range(len(top_list)):
                    self.main_table.insertRow(i+len_total)
                    result=self.get_mainView(top_list[i][0]) # 메인뷰에 띄울것들 호출
                    
                    for j in range(len(result.columns)):
                        item=QTableWidgetItem(result[result.columns[j]][0])
                        item.setTextAlignment(Qt.AlignCenter)
                        self.main_table.setItem(i+len_total,j,item)
                    
                    item=QTableWidgetItem('매수')
                    item.setTextAlignment(Qt.AlignCenter)
                    self.main_table.setItem(i+len_total,8,item)
                    item=QTableWidgetItem(str(top_list[i][1]))
                    item.setTextAlignment(Qt.AlignCenter)
                    self.main_table.setItem(i+len_total,9,item)

                    if top_list[i][2] == 2 :
                        item=QTableWidgetItem('상')
                    elif top_list[i][2] == 1 :
                        item=QTableWidgetItem('중')
                    else:
                        item=QTableWidgetItem('하')
                    item.setTextAlignment(Qt.AlignCenter)
                    self.main_table.setItem(i+len_total,10,item)

                    item=QTableWidgetItem('거래상위')
                    item.setTextAlignment(Qt.AlignCenter)
                    self.main_table.setItem(i+len_total,11,item)

        elif optVal==3:
            global inc_list
            inc_list=self.db.get_buy_list_crawl(3)

            # 메인화면에 거래증가 띄우기
            if len(inc_list) > 0:
        
                for i in range(len(inc_list)):
                    self.main_table.insertRow(i+len_total)
                    result=self.get_mainView(inc_list[i][0]) # 메인뷰에 띄울것들 호출
                    
                    for j in range(len(result.columns)):
                        item=QTableWidgetItem(result[result.columns[j]][0])
                        item.setTextAlignment(Qt.AlignCenter)
                        self.main_table.setItem(i+len_total,j,item)
                    
                    item=QTableWidgetItem('매수')
                    item.setTextAlignment(Qt.AlignCenter)
                    self.main_table.setItem(i+len_total,8,item)
                    item=QTableWidgetItem(str(inc_list[i][1]))
                    item.setTextAlignment(Qt.AlignCenter)
                    self.main_table.setItem(i+len_total,9,item)

                    if inc_list[i][2] == 2 :
                        item=QTableWidgetItem('상')
                    elif inc_list[i][2] == 1 :
                        item=QTableWidgetItem('중')
                    else:
                        item=QTableWidgetItem('하')
                    item.setTextAlignment(Qt.AlignCenter)
                    self.main_table.setItem(i+len_total,10,item)

                    item=QTableWidgetItem('거래증가')
                    item.setTextAlignment(Qt.AlignCenter)
                    self.main_table.setItem(i+len_total,11,item)

        elif optVal==4:
            global dec_list
            dec_list=self.db.get_buy_list_crawl(4)

            # 메인화면에 급락 띄우기
            if len(dec_list) > 0:
        
                for i in range(len(dec_list)):
                    self.main_table.insertRow(i+len_total)
                    result=self.get_mainView(dec_list[i][0]) # 메인뷰에 띄울것들 호출
                    
                    for j in range(len(result.columns)):
                        item=QTableWidgetItem(result[result.columns[j]][0])
                        item.setTextAlignment(Qt.AlignCenter)
                        self.main_table.setItem(i+len_total,j,item)
                    
                    item=QTableWidgetItem('매수')
                    item.setTextAlignment(Qt.AlignCenter)
                    self.main_table.setItem(i+len_total,8,item)
                    item=QTableWidgetItem(str(dec_list[i][1]))
                    item.setTextAlignment(Qt.AlignCenter)
                    self.main_table.setItem(i+len_total,9,item)

                    if dec_list[i][2] == 2 :
                        item=QTableWidgetItem('상')
                    elif dec_list[i][2] == 1 :
                        item=QTableWidgetItem('중')
                    else:
                        item=QTableWidgetItem('하')
                    item.setTextAlignment(Qt.AlignCenter)
                    self.main_table.setItem(i+len_total,10,item)

                    item=QTableWidgetItem('거래감소')
                    item.setTextAlignment(Qt.AlignCenter)
                    self.main_table.setItem(i+len_total,11,item)


    
    def groupboxRadFunction(self):
        '''
        옵션값  [급등:0 ,급락:1 ,거래상위:2 ,거래증가:3 ,거래감소:4]
        '''
        global optVal
        
        if self.rapidUp_rd.isChecked():
            optVal=0
        elif self.rapidDown_rd.isChecked():
            optVal=1
        elif self.tradeRank_rd.isChecked():
            optVal=2
        elif self.tradeASC_rd.isChecked():
            optVal=3
        elif self.tradeDESC_rd.isChecked():
            optVal=4

    def renew_clicked(self):
        nowTime=datetime.datetime.now().strftime("%H:%M %p")
        self.lbl_criteria.setText(nowTime)
        self.main_table.setRowCount(len_total)

        # 메인화면에 보유종목(sell) 띄우기
        if len_sell_retain > 0:

            for i in range(len_sell_retain):
                result=self.get_mainView(sell_retain[i][0]) # 메인뷰에 띄울것들 호출
                
                for j in range(len(result.columns)):
                    item=QTableWidgetItem(result[result.columns[j]][0])
                    item.setTextAlignment(Qt.AlignCenter)
                    self.main_table.setItem(i,j,item)
                
                item=QTableWidgetItem('매도')
                item.setTextAlignment(Qt.AlignCenter)
                self.main_table.setItem(i,8,item)
                item=QTableWidgetItem(str(sell_retain[i][1]))
                item.setTextAlignment(Qt.AlignCenter)
                self.main_table.setItem(i,9,item)

                if sell_retain[i][2] == 2 :
                    item=QTableWidgetItem('상')
                elif sell_retain[i][2] == 1 :
                    item=QTableWidgetItem('중')
                else:
                    item=QTableWidgetItem('하')
                item.setTextAlignment(Qt.AlignCenter)
                self.main_table.setItem(i,10,item)

                item=QTableWidgetItem('보유')
                item.setTextAlignment(Qt.AlignCenter)
                self.main_table.setItem(i,11,item)

        # 메인화면에 보유종목(buy) 띄우기
        if len_buy_retain > 0:
    
            for i in range(len_buy_retain):
                result=self.get_mainView(buy_retain[i][0]) # 메인뷰에 띄울것들 호출
                
                for j in range(len(result.columns)):
                    item=QTableWidgetItem(result[result.columns[j]][0])
                    item.setTextAlignment(Qt.AlignCenter)
                    self.main_table.setItem(i+len_sell_retain,j,item)
                
                item=QTableWidgetItem('매수')
                item.setTextAlignment(Qt.AlignCenter)
                self.main_table.setItem(i+len_sell_retain,8,item)
                item=QTableWidgetItem(str(buy_retain[i][1]))
                item.setTextAlignment(Qt.AlignCenter)
                self.main_table.setItem(i+len_sell_retain,9,item)

                if buy_retain[i][2] == 2 :
                    item=QTableWidgetItem('상')
                elif buy_retain[i][2] == 1 :
                    item=QTableWidgetItem('중')
                else:
                    item=QTableWidgetItem('하')
                item.setTextAlignment(Qt.AlignCenter)
                self.main_table.setItem(i+len_sell_retain,10,item)

                item=QTableWidgetItem('보유')
                item.setTextAlignment(Qt.AlignCenter)
                self.main_table.setItem(i+len_sell_retain,11,item)

        # 메인화면에 관심종목(buy) 띄우기
        if len_buy_interest > 0:
    
            for i in range(len_buy_interest):
                result=self.get_mainView(buy_interest[i][0]) # 메인뷰에 띄울것들 호출
                
                for j in range(len(result.columns)):
                    item=QTableWidgetItem(result[result.columns[j]][0])
                    item.setTextAlignment(Qt.AlignCenter)
                    self.main_table.setItem(i+len_sell_retain+len_buy_retain,j,item)
                
                item=QTableWidgetItem('매수')
                item.setTextAlignment(Qt.AlignCenter)
                self.main_table.setItem(i+len_sell_retain+len_buy_retain,8,item)
                item=QTableWidgetItem(str(buy_interest[i][1]))
                item.setTextAlignment(Qt.AlignCenter)
                self.main_table.setItem(i+len_sell_retain+len_buy_retain,9,item)

                if buy_interest[i][2] == 2 :
                    item=QTableWidgetItem('상')
                elif buy_interest[i][2] == 1 :
                    item=QTableWidgetItem('중')
                else:
                    item=QTableWidgetItem('하')
                item.setTextAlignment(Qt.AlignCenter)
                self.main_table.setItem(i+len_sell_retain+len_buy_retain,10,item)

                item=QTableWidgetItem('관심')
                item.setTextAlignment(Qt.AlignCenter)
                self.main_table.setItem(i+len_sell_retain+len_buy_retain,11,item)

        if optVal==0:
            up_list=self.db.get_buy_list_crawl(0)

            # 메인화면에 급등 띄우기
            if len(up_list) > 0:
        
                for i in range(len(up_list)):
                    self.main_table.insertRow(i+len_total)
                    result=self.get_mainView(up_list[i][0]) # 메인뷰에 띄울것들 호출
                    
                    for j in range(len(result.columns)):
                        item=QTableWidgetItem(result[result.columns[j]][0])
                        item.setTextAlignment(Qt.AlignCenter)
                        self.main_table.setItem(i+len_total,j,item)
                    
                    item=QTableWidgetItem('매수')
                    item.setTextAlignment(Qt.AlignCenter)
                    self.main_table.setItem(i+len_total,8,item)
                    item=QTableWidgetItem(str(up_list[i][1]))
                    item.setTextAlignment(Qt.AlignCenter)
                    self.main_table.setItem(i+len_total,9,item)

                    if up_list[i][2] == 2 :
                        item=QTableWidgetItem('상')
                    elif up_list[i][2] == 1 :
                        item=QTableWidgetItem('중')
                    else:
                        item=QTableWidgetItem('하')
                    item.setTextAlignment(Qt.AlignCenter)
                    self.main_table.setItem(i+len_total,10,item)

                    item=QTableWidgetItem('급등')
                    item.setTextAlignment(Qt.AlignCenter)
                    self.main_table.setItem(i+len_total,11,item)
                
            self.main_table.resizeRowsToContents()
            self.main_table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        elif optVal==1:
            down_list=self.db.get_buy_list_crawl(1)

            # 메인화면에 급락 띄우기
            if len(down_list) > 0:
        
                for i in range(len(down_list)):
                    self.main_table.insertRow(i+len_total)
                    result=self.get_mainView(down_list[i][0]) # 메인뷰에 띄울것들 호출
                    
                    for j in range(len(result.columns)):
                        item=QTableWidgetItem(result[result.columns[j]][0])
                        item.setTextAlignment(Qt.AlignCenter)
                        self.main_table.setItem(i+len_total,j,item)
                    
                    item=QTableWidgetItem('매수')
                    item.setTextAlignment(Qt.AlignCenter)
                    self.main_table.setItem(i+len_total,8,item)
                    item=QTableWidgetItem(str(down_list[i][1]))
                    item.setTextAlignment(Qt.AlignCenter)
                    self.main_table.setItem(i+len_total,9,item)

                    if down_list[i][2] == 2 :
                        item=QTableWidgetItem('상')
                    elif down_list[i][2] == 1 :
                        item=QTableWidgetItem('중')
                    else:
                        item=QTableWidgetItem('하')
                    item.setTextAlignment(Qt.AlignCenter)
                    self.main_table.setItem(i+len_total,10,item)

                    item=QTableWidgetItem('급락')
                    item.setTextAlignment(Qt.AlignCenter)
                    self.main_table.setItem(i+len_total,11,item)
        
        elif optVal==2:
            top_list=self.db.get_buy_list_crawl(2)

            # 메인화면에 거래상위 띄우기
            if len(top_list) > 0:
        
                for i in range(len(top_list)):
                    self.main_table.insertRow(i+len_total)
                    result=self.get_mainView(top_list[i][0]) # 메인뷰에 띄울것들 호출
                    
                    for j in range(len(result.columns)):
                        item=QTableWidgetItem(result[result.columns[j]][0])
                        item.setTextAlignment(Qt.AlignCenter)
                        self.main_table.setItem(i+len_total,j,item)
                    
                    item=QTableWidgetItem('매수')
                    item.setTextAlignment(Qt.AlignCenter)
                    self.main_table.setItem(i+len_total,8,item)
                    item=QTableWidgetItem(str(top_list[i][1]))
                    item.setTextAlignment(Qt.AlignCenter)
                    self.main_table.setItem(i+len_total,9,item)

                    if top_list[i][2] == 2 :
                        item=QTableWidgetItem('상')
                    elif top_list[i][2] == 1 :
                        item=QTableWidgetItem('중')
                    else:
                        item=QTableWidgetItem('하')
                    item.setTextAlignment(Qt.AlignCenter)
                    self.main_table.setItem(i+len_total,10,item)

                    item=QTableWidgetItem('거래상위')
                    item.setTextAlignment(Qt.AlignCenter)
                    self.main_table.setItem(i+len_total,11,item)

        elif optVal==3:
            inc_list=self.db.get_buy_list_crawl(3)

            # 메인화면에 거래증가 띄우기
            if len(inc_list) > 0:
        
                for i in range(len(inc_list)):
                    self.main_table.insertRow(i+len_total)
                    result=self.get_mainView(inc_list[i][0]) # 메인뷰에 띄울것들 호출
                    
                    for j in range(len(result.columns)):
                        item=QTableWidgetItem(result[result.columns[j]][0])
                        item.setTextAlignment(Qt.AlignCenter)
                        self.main_table.setItem(i+len_total,j,item)
                    
                    item=QTableWidgetItem('매수')
                    item.setTextAlignment(Qt.AlignCenter)
                    self.main_table.setItem(i+len_total,8,item)
                    item=QTableWidgetItem(str(inc_list[i][1]))
                    item.setTextAlignment(Qt.AlignCenter)
                    self.main_table.setItem(i+len_total,9,item)

                    if inc_list[i][2] == 2 :
                        item=QTableWidgetItem('상')
                    elif inc_list[i][2] == 1 :
                        item=QTableWidgetItem('중')
                    else:
                        item=QTableWidgetItem('하')
                    item.setTextAlignment(Qt.AlignCenter)
                    self.main_table.setItem(i+len_total,10,item)

                    item=QTableWidgetItem('거래증가')
                    item.setTextAlignment(Qt.AlignCenter)
                    self.main_table.setItem(i+len_total,11,item)

        elif optVal==4:
            dec_list=self.db.get_buy_list_crawl(4)

            # 메인화면에 급락 띄우기
            if len(dec_list) > 0:
        
                for i in range(len(dec_list)):
                    self.main_table.insertRow(i+len_total)
                    result=self.get_mainView(dec_list[i][0]) # 메인뷰에 띄울것들 호출
                    
                    for j in range(len(result.columns)):
                        item=QTableWidgetItem(result[result.columns[j]][0])
                        item.setTextAlignment(Qt.AlignCenter)
                        self.main_table.setItem(i+len_total,j,item)
                    
                    item=QTableWidgetItem('매수')
                    item.setTextAlignment(Qt.AlignCenter)
                    self.main_table.setItem(i+len_total,8,item)
                    item=QTableWidgetItem(str(dec_list[i][1]))
                    item.setTextAlignment(Qt.AlignCenter)
                    self.main_table.setItem(i+len_total,9,item)

                    if dec_list[i][2] == 2 :
                        item=QTableWidgetItem('상')
                    elif dec_list[i][2] == 1 :
                        item=QTableWidgetItem('중')
                    else:
                        item=QTableWidgetItem('하')
                    item.setTextAlignment(Qt.AlignCenter)
                    self.main_table.setItem(i+len_total,10,item)

                    item=QTableWidgetItem('거래감소')
                    item.setTextAlignment(Qt.AlignCenter)
                    self.main_table.setItem(i+len_total,11,item)

        self.main_table.resizeRowsToContents()
        self.main_table.setEditTriggers(QAbstractItemView.NoEditTriggers)   

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