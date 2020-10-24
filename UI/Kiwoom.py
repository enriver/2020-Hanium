'''
"python.linting.pylintArgs": [
    "--extension-pkg-whitelist=PyQt5"
    ]


'''

import sys
from PyQt5.QtWidgets import *
from PyQt5.QAxContainer import *
from PyQt5.QtCore import *
import time

TR_REQ_TIME_INTERVAL=0.2

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

    def _get_repeat_cnt(self,trcode,rqname):
        ret=self.dynamicCall("GetRepeatCnt(QString,QString)",trcode,rqname)
        return ret

    def _receive_tr_data(self, screen_no, rqname, trcode, record_name, next, un1, un2, un3, un4):
        if next == '2':
            self.remained_data=True
        else:
            self.remained_data=False

        if rqname=='opw00001_req':
            self._opw00001(rqname,trcode)
        elif rqname=='opw00018_req':
            self._opw00018(rqname,trcode)
        elif rqname=='opt10081_req':
            self._opt10081(rqname,trcode)
        elif rqname=='opt10001_req':
            self._opt10001(rqname,trcode)
        elif rqname=='mainView':
            self._optMainView(rqname,trcode)

        try:
            self.tr_event_loop.exit()
        except AttributeError:
            pass

    def _opw00001(self, rqname,trcode):
        d2_deposit=self._comm_get_data(trcode,"",rqname,0,"d+2 추정예수금")
        self.d2_deposit=Kiwoom.change_format(d2_deposit)
    
    def reset_opw00018_output(self):
        self.opw00018_output={'single':[],'multi':[],'retained':[]}
        
    def _opw00018(self,rqname,trcode):

        # 전체
        total_purchase_price = self._comm_get_data(trcode, "", rqname, 0, "총매입금액")
        total_eval_price = self._comm_get_data(trcode, "", rqname, 0, "총평가금액")
        total_eval_profit_loss_price = self._comm_get_data(trcode, "", rqname, 0, "총평가손익금액")
        total_earning_rate = self._comm_get_data(trcode, "", rqname, 0, "총수익률(%)")
        estimated_deposit = self._comm_get_data(trcode, "", rqname, 0, "추정예탁자산")

        total_earning_rate=Kiwoom.change_format2(total_earning_rate)

        '''
        if self.get_server_gubun():
            total_earning_rate=float(total_earning_rate)/100
            total_earning_rate=str(total_earning_rate)
        '''
        self.opw00018_output['single'].append(Kiwoom.change_format(total_purchase_price))
        self.opw00018_output['single'].append(Kiwoom.change_format(total_eval_price))
        self.opw00018_output['single'].append(Kiwoom.change_format(total_eval_profit_loss_price))
        self.opw00018_output['single'].append(total_earning_rate)
        self.opw00018_output['single'].append(Kiwoom.change_format(estimated_deposit))

        # 종목별
        rows=self._get_repeat_cnt(trcode,rqname)

        for i in range(rows):
            name=self._comm_get_data(trcode,"",rqname,i,"종목명")
            quantity=self._comm_get_data(trcode,"",rqname,i,"보유수량")
            purchase_price=self._comm_get_data(trcode,"",rqname,i,"매입가")
            current_price=self._comm_get_data(trcode,"",rqname,i,"현재가")
            eval_profit_loss_price=self._comm_get_data(trcode,"",rqname,i,"평가손익")
            earning_rate=self._comm_get_data(trcode,"",rqname,i,"수익률(%)")
            code=self._comm_get_data(trcode,"",rqname,i,"종목번호")

            quantity=Kiwoom.change_format(quantity)
            purchase_price=Kiwoom.change_format(purchase_price)
            current_price=Kiwoom.change_format(current_price)
            eval_profit_loss_price=Kiwoom.change_format(eval_profit_loss_price)
            earning_rate=Kiwoom.change_format2(earning_rate)

            self.opw00018_output['multi'].append([name,quantity,purchase_price,current_price,eval_profit_loss_price,earning_rate])
            self.opw00018_output['retained'].append([code,name,quantity])

    def _opt10081(self,rqname,trcode):
        data_cnt=self._get_repeat_cnt(trcode,rqname)

        for i in range(data_cnt):
            date=self._comm_get_data(trcode,"",rqname,i,"일자")
            open=self._comm_get_data(trcode,"",rqname,i,"시가")
            high=self._comm_get_data(trcode,"",rqname,i,"고가")
            low=self._comm_get_data(trcode,"",rqname,i,"저가")
            close=self._comm_get_data(trcode,"",rqname,i,"현재가")
            volume=self._comm_get_data(trcode,"",rqname,i,"거래량")

            self.ohlcv['date'].append(date)
            self.ohlcv['open'].append(int(open))
            self.ohlcv['high'].append(int(high))
            self.ohlcv['low'].append(int(low))
            self.ohlcv['close'].append(int(close))
            self.ohlcv['volume'].append(int(volume))

    def _opt10001(self,rqname,trcode):
        nowVal=self.dynamicCall("CommGetData(QString,QString,QString,int,QString)",trcode,"",rqname,0,"현재가").strip()
        diff=self.dynamicCall("CommGetData(QString,QString,QString,int,QString)",trcode,"",rqname,0,"전일대비").strip()
        rate=self.dynamicCall("CommGetData(QString,QString,QString,int,QString)",trcode,"",rqname,0,"등락율").strip()
        quant=self.dynamicCall("CommGetData(QString,QString,QString,int,QString)",trcode,"",rqname,0,"거래량").strip()
        open=self.dynamicCall("CommGetData(QString,QString,QString,int,QString)",trcode,"",rqname,0,"시가").strip()
        high=self.dynamicCall("CommGetData(QString,QString,QString,int,QString)",trcode,"",rqname,0,"고가").strip()
        low=self.dynamicCall("CommGetData(QString,QString,QString,int,QString)",trcode,"",rqname,0,"저가").strip()
        up=self.dynamicCall("CommGetData(QString,QString,QString,int,QString)",trcode,"",rqname,0,"상한가").strip()
        down=self.dynamicCall("CommGetData(QString,QString,QString,int,QString)",trcode,"",rqname,0,"하한가").strip()
        per=self.dynamicCall("CommGetData(QString,QString,QString,int,QString)",trcode,"",rqname,0,"PER").strip()
        roe=self.dynamicCall("CommGetData(QString,QString,QString,int,QString)",trcode,"",rqname,0,"ROE").strip()

        nowVal=int(float(nowVal[1:]))
        quant=Kiwoom.change_format(quant)
        open=Kiwoom.change_format(open[1:])
        high=Kiwoom.change_format(high[1:])
        low=Kiwoom.change_format(low[1:])
        up=Kiwoom.change_format(up[1:])
        down=Kiwoom.change_format(down[1:])

        if diff=='0':
            diff=int(float(diff))
            self.searchView['diff'].append(diff)
        else:
            diff=int(float(diff[1:]))
            self.searchView['diff'].append(diff)
    
        self.searchView['nowVal'].append(nowVal)
        self.searchView['rate'].append(rate+"%")
        self.searchView['quant'].append(quant)
        self.searchView['open'].append(open)
        self.searchView['high'].append(high)
        self.searchView['low'].append(low)
        self.searchView['up'].append(up)
        self.searchView['down'].append(down)
        self.searchView['per'].append(per)
        self.searchView['roe'].append(roe+"%")
        

    def _optMainView(self,rqname,trcode):
        name=self.dynamicCall("CommGetData(QString,QString,QString,int,QString)",trcode,"",rqname,0,"종목명").strip()
        nowVal=self.dynamicCall("CommGetData(QString,QString,QString,int,QString)",trcode,"",rqname,0,"현재가").strip()
        diff=self.dynamicCall("CommGetData(QString,QString,QString,int,QString)",trcode,"",rqname,0,"전일대비").strip()
        rate=self.dynamicCall("CommGetData(QString,QString,QString,int,QString)",trcode,"",rqname,0,"등락율").strip()
        quant=self.dynamicCall("CommGetData(QString,QString,QString,int,QString)",trcode,"",rqname,0,"거래량").strip()
        open=self.dynamicCall("CommGetData(QString,QString,QString,int,QString)",trcode,"",rqname,0,"시가").strip()
        high=self.dynamicCall("CommGetData(QString,QString,QString,int,QString)",trcode,"",rqname,0,"고가").strip()
        low=self.dynamicCall("CommGetData(QString,QString,QString,int,QString)",trcode,"",rqname,0,"저가").strip()
        
        nowVal=Kiwoom.change_format(nowVal[1:])
        diff=Kiwoom.change_format(diff)
        quant=Kiwoom.change_format(quant)
        open=Kiwoom.change_format(open[1:])
        high=Kiwoom.change_format(high[1:])
        low=Kiwoom.change_format(low[1:])

        self.mainView['name'].append(name)
        self.mainView['nowVal'].append(nowVal)
        self.mainView['diff'].append(diff)
        self.mainView['rate'].append(rate+"%")
        self.mainView['quant'].append(quant)
        self.mainView['open'].append(open)
        self.mainView['high'].append(high)
        self.mainView['low'].append(low)
        

    def get_server_gubun(self):
        ret=self.dynamicCall("KOA_Functions(QString,QString)","GetServerGubun","")
        return ret

    def send_order(self, rqname, screen_no, acc_no, order_type, code, quantity, price, hoga, order_no):
        self.dynamicCall("SendOrder(QString, QString, QString, int, QString, int, int, QString, QString)",[rqname, screen_no, acc_no, order_type, code, quantity, price, hoga, order_no])

    

    @staticmethod
    def change_format(data):
        strip_data=data.lstrip('-0')
        if strip_data=='':  
            strip_data='0'

        try:
            format_data=format(int(strip_data),',d')
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
