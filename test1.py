from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog
import sys

class Example(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setGeometry(100, 100, 300, 200)
        self.setWindowTitle('Example')

        btn = QPushButton('选择文件', self)
        btn.move(50, 50)
        btn.clicked.connect(self.selectFile)

    def selectFile(self):
        filename, _ = QFileDialog.getOpenFileName(self, "选择文件", r'C:\Users\123\Desktop\code\python\tensorflow-minist\png', "Image Files (*.jpg *.png)")
        if filename:
            print("选择的文件是：", filename)
            # 在这里可以进行文件路径的处理和后续的逻辑操作

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    ex.show()
    sys.exit(app.exec_())