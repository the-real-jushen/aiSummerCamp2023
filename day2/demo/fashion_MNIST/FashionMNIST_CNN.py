# %%
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

tf.__version__
# %%
fashion_mnist = tf.keras.datasets.fashion_mnist
# 这里会把数据下载到用户目录下~\.keras\datasets中，如果下载中断需要手动清一下resids
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# %%
# 共10类图片，0-9如下：
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# %%
# 训练集包含60000张28*28的图片
train_images.shape
# %%
# 标签0-9，60000张图一共60000个没啥说的
len(train_labels)
# %%
train_labels
# %%
# 画张图看看，像素的范围为[0, 255]
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
# %%
# 像素归一化，归一化后的像素范围为[0, 1]
train_images = train_images / 255.0

test_images = test_images / 255.0
# %%
# 肉眼可见图变灰了
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
# %%
# 进行一个极其简单的神经网络的搭建
# Flatten层把二维的输入转为一维向量
# Dense层即全连接层，第一层Dense作为隐藏层，激活函数为relu
# 最后一层Dense作为输出，输出每个类别的概率
cnn_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10)
])
# %%
# 编译模型
# 优化器选择adam（不知道选啥的时候用adam就完事了）
# 损失函数选用SparseCategoricalCrossentropy
cnn_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
# %%
# fit模型，开始训练啦！
cnn_model.fit(train_images, train_labels, epochs=10, batch_size=32)
# %%
# 测试模型
test_loss, test_acc = cnn_model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)
# %%
# 这里直接接一个softmax把输出转化成概率
probability_model = tf.keras.Sequential([cnn_model, 
                                         tf.keras.layers.Softmax()])
# %%
predictions = probability_model.predict(test_images)
# %%
predictions[0]
# %%
# argmax把输出最大概率的元素，得到结果
np.argmax(predictions[0])
# %%
test_labels[0]
# %%
# 下面都是可视化，感兴趣的自己看看
def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
# %%
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()
# %%
i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()
# %%
# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()
# %%
