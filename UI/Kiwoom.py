import sys
from PyQt5.QtWidgets import *
from PyQt5.QAxContainer import *
from PyQt5.QtCore import *
import time


class Kiwoom(QAxWidget):
    def __init__(self):
        super().__init__()
        self._create_kiwoom_instance()
        self._set_signal_slots()

    def _create_kiwoom_instance(self):
        self.setControl("KHOPENAPI.KHOpenAPICtrl.1")

    def _set_signal_slots(self):
        self.OnEventConnect.connect(self._event_connect)
        self.OnReceiveTrData.connect(self._receive_tr_data)

    def comm_rq_data(self,rqname,trcode,next,screen_no):
        self.dynamicCall("CommRqData(QString,QString,int,QString)", rqname, trcode, next, screen_no)
        self.tr_event_loop=QEventLoop()
        self.tr_event_loop.exec_()

    def comm_connect(self):
        self.dynamicCall("CommConnect()")
        self.login_event_loop = QEventLoop()
        self.login_event_loop.exec_()

    def _event_connect(self, err_code):
        if err_code == 0:

            print("connected")
          
        else:
            print("disconnected")
            self.close()

        self.login_event_loop.exit()
    
    def set_input_value(self,id,value):
        self.dynamicCall("SetInputValue(QString,QString)",id,value)

    def get_connect_state(self):
        ret = self.dynamicCall("GetConnectState()")
        return ret

    def _comm_get_data(self,code,real_type,field_name,index,item_name):
        ret=self.dynamicCall("CommGetData(QString,QString,QString,int,QString)",code,real_type,field_name,index,item_name)

        return ret.strip()

    def _receive_tr_data(self, screen_no, rqname, trcode, record_name, next, un1, un2, un3, un4):
        if next == '2':
            self.remained_data=True
        else:
            self.remained_data=False

        if rqname=='opw00001_req':
            self._opw00001(rqname,trcode)
        elif rqname=='opw00018_req':
            self._opw00018(rqname,trcode)

        try:
            self.tr_event_loop.exit()
        except AttributeError:
            pass

    def _opw00001(self, rqname,trcode):
        d2_deposit=self._comm_get_data(trcode,"",rqname,0,"d+2 추정예수금")
        self.d2_deposit=Kiwoom.change_format(d2_deposit)
    
    def _opw00018(self,rqname,trcode):
        total_purchase_price = self._comm_get_data(trcode, "", rqname, 0, "총매입금액")
        total_eval_price = self._comm_get_data(trcode, "", rqname, 0, "총평가금액")
        total_eval_profit_loss_price = self._comm_get_data(trcode, "", rqname, 0, "총평가손익금액")
        total_earning_rate = self._comm_get_data(trcode, "", rqname, 0, "총수익률(%)")
        estimated_deposit = self._comm_get_data(trcode, "", rqname, 0, "추정예탁자산")

        self.total_purchase_price=Kiwoom.change_format(total_purchase_price)
        self.total_eval_price=Kiwoom.change_format(total_eval_price)
        self.total_eval_profit_loss_price=Kiwoom.change_format(total_eval_profit_loss_price)
        self.total_earning_rate=Kiwoom.change_format(total_earning_rate)
        self.estimated_deposit=Kiwoom.change_format(estimated_deposit)

    @staticmethod
    def change_format(data):
        strip_data=data.lstrip('-0')
        if strip_data=='':
            strip_data='0'

        try:
            format_data=foramt(int(strip_data),',d')
        except:
            format_data=format(float(strip_data))

        if data.startswith('-'):
            format_data='-' + format_data

        return format_data

    @staticmethod
    def change_format2(data):
        strip_data=data.lstrip('-0')

        if strip_data == '':
            strip_data='0'
        
        if strip_data.startswith('.'):
            strip_data='0'+strip_data

        if data.startswith('-'):
            strip_data='-'+strip_data

        return strip_data
