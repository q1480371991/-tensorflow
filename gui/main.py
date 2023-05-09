'''
Created on 2018-08-09 00:00

@author: Freedom
'''

from MainWidget import MainWidget
from PyQt5.QtWidgets import QApplication

import sys


def main():
    app = QApplication(sys.argv)

    mainWidget = MainWidget()  # 新建一个主界面
    mainWidget.show()  # 显示主界面

    exit(app.exec_())  # 进入消息循环


if __name__ == '__main__':
    main()
