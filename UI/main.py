# 메인 UI

import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QAxContainer import *

class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("존나하기싫다")
        self.setGeometry(300,300,300,400)

        self.kiwoom=QAxWidget("KHOPENAPI.KHOpenAPICtrl.1")

        btn1=QPushButton("Login",self)
        btn1.move(20,20)
        btn1.clicked.connect(self.btn1_clicked)

        btn2=QPushButton("Check State",self)
        btn2.move(20,70)
        btn2.clicked.connect(self.btn2_clicked)

    def btn1_clicked(self):
        ret=self.kiwoom.dynamicCall("CommConnect()")

    def btn2_clicked(self):
        if self.kiwoom.dynamicCall("GetConnected()")==0:
            self.statusBar().showMessage("Not connected")
        else:
            self.statusBar().showMessage("Connected")

if __name__=="__main__":
    app=QApplication(sys.argv)
    mywindow=MyWindow()
    mywindow.show()
    app.exec_()