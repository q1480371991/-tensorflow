import tensorflow as tf
import os
from getmyds import getds
from matplotlib import pyplot as plt
#mnist数据集
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
#图片数据集
# x_train, y_train, x_test, y_test = getds()

'''
当我们给计算机一个手写数字图片，计算机通常会把这个图片转化为数字矩阵的形式，即将黑白灰等颜色映射到数字上，比如黑色为0，白色为1。
接下来，我们需要从这个数字矩阵中提取特征向量，这个特征向量就是对输入数据的精简表示，尽可能地保留了有用的信息。
对于手写数字识别任务，我们可以将一个数字图片的像素点构成的数字矩阵看成一个特征向量，比如一个28×28的数字图片矩阵就可以转化为一个784784维的特征向量。
接着，我们需要通过机器学习算法对这些特征向量进行分类，将它们分为数字0到9中的某一个。
'''

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),#flatten层，将输入数据展平成一维向量
    tf.keras.layers.Dense(128, activation='relu'),#128个神经元的全连接层，将输入的特征向量与一组学习的权重相乘，产生一个隐藏向量，捕捉输入数据的特征。
    tf.keras.layers.Dense(10, activation='softmax')#10 个神经元的全连接层，该层将使用训练数据来学习其权重，以使每个神经元可以对输入数据进行分类，并输出每个类别的概率得分。
])
#使用amam优化器来优化模型的权重参数
#使用稀疏分类交叉熵作为损失函数，用来优化多分类问题的损失函数
#使用 sparse_categorical_accuracy 作为度量指标，用来度量分类问题上的精度。
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = "./model/nummodel.ckpt"
#断点续训
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

#在训练过程中保存模型的检查点，并保存模型最好的权重
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)

#训练模型，迭代20次，训练1次后检验，每一次epoch结束后保存模型参数
#history对象记录了训练过程中每个epoch的训练指标和测试指标，用于可视化模型的训练过程
history = model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data=(x_test, y_test), validation_freq=1,
                    callbacks=[cp_callback])
#打印出模型的结构和参数数量等信息
model.summary()

# 可视化训练集和验证集的acc和loss曲线
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
#将两个图表显示在一行中，第一个图表在第一列，第二个图表在第二列。
plt.subplot(1, 2, 1)
#绘制曲线
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
