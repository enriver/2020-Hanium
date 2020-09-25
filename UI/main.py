# 메인 UI

import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import uic
from Kiwoom import *
import datetime

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
        account_num=self.kiwoom.dynamicCall("GetLoginInfo(QString)",["ACCNO"])
        self.cmb_account.addItem(account_num.rstrip(';'))

        #사용자명 가져오기
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
        self.kiwoom.set_input_value("계좌번호",account_num.rstrip(';'))
        self.kiwoom.set_input_value("비밀번호","0000")
        self.kiwoom.comm_rq_data("opw00001_req","opw00001",0,"2000")
        
        self.lbl_deposit.setText(self.kiwoom.d2_deposit)

        #추정자산,총매입금,총평가금,손익,수익률
        self.kiwoom.reset_opw00018_output()
        self.kiwoom.set_input_value("계좌번호",account_num.rstrip(';'))
        self.kiwoom.comm_rq_data("opw00018_req","opw00018",0,"2000")

        self.lbl_asset.setText(self.kiwoom.opw00018_output['single'][4])
        self.lbl_eval_amt.setText(self.kiwoom.opw00018_output['single'][1])
        self.lbl_purchase.setText(self.kiwoom.opw00018_output['single'][0])
        self.lbl_profitLoss.setText(self.kiwoom.opw00018_output['single'][2])
        self.lbl_ror.setText(self.kiwoom.opw00018_output['single'][3])

        #종목 자동완성
        code_list=self.kiwoom.dynamicCall("GetCodeListByMarket(QString)",["0"])
        kospi_code_list=code_list.split(';')
        kospi_code_name_list=[]

        for x in kospi_code_list:
            name=self.kiwoom.dynamicCall("GetMasterCodeName(QString)",[x])
            kospi_code_name_list.append(name)
        
        name_completer=QCompleter(kospi_code_name_list)
        self.lineEdit.setCompleter(name_completer)

        code_completer=QCompleter(kospi_code_list)
        self.lineEdit_2.setCompleter(code_completer)
    
    def optSave_clicked(self):
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