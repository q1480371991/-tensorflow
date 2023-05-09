import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import re

from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog


class DigitalRecognition():
    def __init__(self):
        self.model_save_path = r'C:\Users\123\Desktop\code\python\tensorflow-minist\model\nummodel.ckpt'
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        status = self.model.load_weights(self.model_save_path)
        status.expect_partial()

    def check(self,image_path):
        searchstr = "_"
        res = re.search(searchstr, image_path)
        if res != None:
            return False
        else:
            return True

    def select(self):
        app = QApplication([])
        widget = QWidget()
        filename = QFileDialog.getOpenFileName(widget, "选择文件", "../png/", "Image Files (*.jpg *.png)")
        # print(filename[0])
        app.quit()
        num=self.predict(filename[0])
        return num
    def predict(self,image_path):
        img = Image.open(image_path)

        image = plt.imread(image_path)


        img = img.resize((28, 28), Image.ANTIALIAS)

        # 将图片从彩色转换为弧度图像PIL，再将PIL图像转换为Numpy数组
        img_arr = np.array(img.convert('L'))
        plt.set_cmap('gray')
        plt.imshow(img_arr)
        # 由于训练集是黑底白字，所以转图片转换为黑底白字
        # 颜色取反：img_arr=255-img_arr
        # 让输入图片变成只有黑色和白色的高对比度图
        if self.check(image_path):
            for i in range(28):
                for j in range(28):
                    if img_arr[i][j] < 200:
                        # 纯白色
                        img_arr[i][j] = 255
                    else:
                        # 纯黑色
                        img_arr[i][j] = 0
        # 归一化
        img_arr = img_arr / 255.0# 归一化处理
        x_predict = img_arr[tf.newaxis, ...]# 转化为张量作为输入
        result = self.model.predict(x_predict)# 得到预测结果
        # 返回的pre是张量
        pred = tf.argmax(result, axis=1)# 解码得到预测数字

        print("图片", image_path, "的数字为:", end="")
        # tf.print(pred)
        # print(pred)

        # 获取张量
        x = pred.numpy()
        print(x)
        plt.pause(1)
        plt.close()
        return x[0]


