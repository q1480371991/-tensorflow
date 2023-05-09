'''
Created on 2018年8月8日
@author: Freedom
'''
from PyQt5.Qt import QWidget, QColor, QPixmap, QIcon, QSize, QCheckBox
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QPushButton, QSplitter, \
    QComboBox, QLabel, QSpinBox, QFileDialog, QTextEdit, QMessageBox, QApplication
from PaintBoard import PaintBoard
import numpy as np
from PIL import Image
import test1
import dr


class MainWidget(QWidget):

    def __init__(self, Parent=None):
        '''
        Constructor
        '''
        super().__init__(Parent)
        self.mydr=dr.DigitalRecognition()
        self.__InitData()  # 先初始化数据，再初始化界面
        self.__InitView()

    def __InitData(self):
        '''
                  初始化成员变量
        '''
        self.__paintBoard = PaintBoard(self)
        # 获取颜色列表(字符串类型)
        self.__colorList = QColor.colorNames()

    def __InitView(self):
        '''
                  初始化界面
        '''
        self.setFixedSize(800, 640)
        self.setWindowTitle("PaintBoard Example PyQt5")

        # 新建一个水平布局作为本窗体的主布局
        main_layout = QHBoxLayout(self)
        # 设置主布局内边距以及控件间距为10px
        main_layout.setSpacing(10)

        # 在主界面左侧放置画板
        main_layout.addWidget(self.__paintBoard)

        # 新建垂直子布局用于放置按键
        sub_layout = QVBoxLayout()

        # 设置此子布局和内部控件的间距为10px
        sub_layout.setContentsMargins(10, 10, 10, 10)

        self.__btn_Clear = QPushButton("清空画板")
        self.__btn_Clear.setParent(self)  # 设置父对象为本界面
        # 将按键按下信号与画板清空函数相关联
        self.__btn_Clear.clicked.connect(self.Cleartxt)
        self.__btn_Clear.clicked.connect(self.__paintBoard.Clear)
        sub_layout.addWidget(self.__btn_Clear)

        #选择图片
        self.__btn_Select = QPushButton("选择测试图片")
        self.__btn_Select.setParent(self)  # 设置父对象为本界面
        # 将按键按下信号与画板清空函数相关联
        self.__btn_Select.clicked.connect(self.select)
        sub_layout.addWidget(self.__btn_Select)


        self.__btn_yuce = QPushButton("人工智能预测")
        self.__btn_yuce.setParent(self)  # 设置父对象为本界面
        self.__btn_yuce.clicked.connect(lambda:self.yuce())
        sub_layout.addWidget(self.__btn_yuce)


        self.__text_out = QTextEdit(self)
        self.__text_out.setParent(self)
        self.__text_out.setObjectName("预测结果为：")
        sub_layout.addWidget(self.__text_out)


        self.__btn_Quit = QPushButton("退出")
        self.__btn_Quit.setParent(self)  # 设置父对象为本界面
        self.__btn_Quit.clicked.connect(self.Quit)
        sub_layout.addWidget(self.__btn_Quit)

        self.__btn_Save = QPushButton("保存作品")
        self.__btn_Save.setParent(self)
        self.__btn_Save.clicked.connect(self.on_btn_Save_Clicked)
        sub_layout.addWidget(self.__btn_Save)

        self.__cbtn_Eraser = QCheckBox("  使用橡皮擦")
        self.__cbtn_Eraser.setParent(self)
        self.__cbtn_Eraser.clicked.connect(self.on_cbtn_Eraser_clicked)
        sub_layout.addWidget(self.__cbtn_Eraser)

        splitter = QSplitter(self)  # 占位符
        sub_layout.addWidget(splitter)

        self.__label_penThickness = QLabel(self)
        self.__label_penThickness.setText("画笔粗细")
        self.__label_penThickness.setFixedHeight(20)
        sub_layout.addWidget(self.__label_penThickness)

        self.__spinBox_penThickness = QSpinBox(self)
        self.__spinBox_penThickness.setMaximum(40)
        self.__spinBox_penThickness.setMinimum(2)
        self.__spinBox_penThickness.setValue(20)  # 默认粗细为10
        self.__spinBox_penThickness.setSingleStep(2)  # 最小变化值为2
        self.__spinBox_penThickness.valueChanged.connect(
            self.on_PenThicknessChange)  # 关联spinBox值变化信号和函数on_PenThicknessChange
        sub_layout.addWidget(self.__spinBox_penThickness)

        self.__label_penColor = QLabel(self)
        self.__label_penColor.setText("画笔颜色")
        self.__label_penColor.setFixedHeight(20)
        sub_layout.addWidget(self.__label_penColor)

        self.__comboBox_penColor = QComboBox(self)
        self.__fillColorList(self.__comboBox_penColor)  # 用各种颜色填充下拉列表
        self.__comboBox_penColor.currentIndexChanged.connect(
            self.on_PenColorChange)  # 关联下拉列表的当前索引变更信号与函数on_PenColorChange
        sub_layout.addWidget(self.__comboBox_penColor)

        main_layout.addLayout(sub_layout)  # 将子布局加入主布局
    def Cleartxt(self):
        self.__text_out.setText("")

    def select(self):
        filename, _ = QFileDialog.getOpenFileName(self, "选择文件", r'C:\Users\123\Desktop\code\python\tensorflow-minist\png', "Image Files (*.jpg *.png)")
        if filename:
            num=self.mydr.predict(filename)
        res = "人工智能判断为:" + str(num)
        self.__text_out.setText(res)
    def __fillColorList(self, comboBox):

        index_black = 0
        index = 0
        for color in self.__colorList:
            if color == "black":
                index_black = index
            index += 1
            pix = QPixmap(70, 20)
            pix.fill(QColor(color))
            comboBox.addItem(QIcon(pix), None)
            comboBox.setIconSize(QSize(70, 20))
            comboBox.setSizeAdjustPolicy(QComboBox.AdjustToContents)

        comboBox.setCurrentIndex(index_black)

    def on_PenColorChange(self):
        color_index = self.__comboBox_penColor.currentIndex()
        color_str = self.__colorList[color_index]
        self.__paintBoard.ChangePenColor(color_str)

    def on_PenThicknessChange(self):
        penThickness = self.__spinBox_penThickness.value()
        self.__paintBoard.ChangePenThickness(penThickness)

    def on_btn_Save_Clicked(self):
        savePath = QFileDialog.getSaveFileName(self, 'Save Your Paint', '.\\', '*.png')
        print(savePath)
        if savePath[0] == "":
            print("Save cancel")
            return
        image = self.__paintBoard.GetContentAsQImage()
        image.save(savePath[0])

    def on_cbtn_Eraser_clicked(self):
        if self.__cbtn_Eraser.isChecked():
            self.__paintBoard.EraserMode = True  # 进入橡皮擦模式
        else:
            self.__paintBoard.EraserMode = False  # 退出橡皮擦模式

    def Quit(self):
        self.close()


    def yuce(self):
        #标准化图片 获取Y
        savePath = r"C:\Users\123\Desktop\code\python\tensorflow-minist\png\test_.png"
        image = self.__paintBoard.GetContentAsQImage()
        image.save(savePath)

        p=self.mydr.predict(savePath)
        res="人工智能判断为:"+str(p)
        self.__text_out.setText(res)
        # res = QMessageBox.information(self,"人工智能判断为：",str(p),QMessageBox.Yes|QMessageBox.No)
        # res.exec_()