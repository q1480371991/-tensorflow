import os.path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets
from matplotlib import pyplot as plt
#加载minist数据集，分成训练集和测试集，每个样本包含图像和标签
(x, y), (x_val, y_val) = datasets.mnist.load_data()
print('datasets', x.shape, y.shape, x.min(), y.min())
#训练集图像数据归一化到0-1之前
#把输入特征的数值变小更适合神经网络吸收
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.
#构建数据集对象
#使用from_tensor_slices把训练集的输入特征和标签配对打包
db = tf.data.Dataset.from_tensor_slices((x, y))
#批量训练，并行计算一次32个样本、所有数据集迭代10次
db = db.batch(32).repeat(10)
model_path= "model/mymodel.h5"
if os.path.exists(model_path):
    print("load mymodel")
    model = keras.models.load_model(model_path)
else:
    print("new model")
    model = keras.Sequential([
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(10)
    ])
    model.build(input_shape=(None, 28 * 28))
#构建Sequential窗口，一共3层网络，并且前一个网络的输出作为后一个网络的输入


#指定输入大小

#打印出网络的结构和参数量
model.summary()
#optimizers用于更新梯度下降算法参数，0.01为学习率
# optimizer = optimizers.SGD(lr=0.01)
# optimizer = optimizers.SGD(learning_rate=0.01)
optimizer = optimizers.SGD(learning_rate=0.01)

#准备率
acc_meter = keras.metrics.Accuracy()
#创建参数文件
# summary_writer = tf.summary.create_file_writer('C:/really not partition/python/minist')
train_loss_results = []  # 将每轮的loss记录在此列表中，为后续画loss曲线提供数据
test_acc = []  # 将每轮的acc记录在此列表中，为后续画acc曲线提供数据
#循环数据集
for step, (xx, yy) in enumerate(db):
    #上下文
    with tf.GradientTape() as tape:
        #图像样本大小重置(-1, 28*28)
        xx = tf.reshape(xx, (-1, 28*28))
        #获取输出
        out = model(xx)
        #实际标签转为onehot编码
        y_onehot = tf.one_hot(yy, depth=10)
        #计算误差
        loss = tf.square(out-y_onehot)
        loss = tf.reduce_sum(loss/xx.shape[0])

    #更新准备率
    acc_meter.update_state(tf.argmax(out, axis=1), yy)
    #更新梯度参数
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    #参数存储，便于查看曲线图
    # with summary_writer.as_default():
    #     tf.summary.scalar('train-loss', float(loss), step=step)
    #     tf.summary.scalar('test-acc', acc_meter.result().numpy(), step=step)
        #tf.summary.image('val-onebyone-images', val)
    if step % 1000 == 0:
        print(step, 'loss:', float(loss), 'acc:', acc_meter.result().numpy())
        #参数存储，便于查看曲线图
        train_loss_results.append(float(loss))
        test_acc.append(acc_meter.result().numpy())

        acc_meter.reset_states()
    #len(db) // 10表示数据集的迭代次数，当训练达到最后一次迭代时，模型将被保存在指定路径下的model.h5文件中
    if step == (len(db) // 10 - 1):
        model.save('mymodel.h5') 
# 绘制 loss 曲线
plt.title('Loss Function Curve')  # 图片标题
plt.xlabel('Epoch')  # x轴变量名称
plt.ylabel('Loss')  # y轴变量名称
plt.plot(train_loss_results, label="$Loss$")  # 逐点画出trian_loss_results值并连线，连线图标是Loss
plt.legend()  # 画出曲线图标
plt.show()  # 画出图像

# 绘制 Accuracy 曲线
plt.title('Acc Curve')  # 图片标题
plt.xlabel('Epoch')  # x轴变量名称
plt.ylabel('Acc')  # y轴变量名称
plt.plot(test_acc, label="$Accuracy$")  # 逐点画出test_acc值并连线，连线图标是Accuracy
plt.legend()
plt.show()


