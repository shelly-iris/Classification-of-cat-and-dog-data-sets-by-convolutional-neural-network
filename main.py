#导入相应的库
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, Model, Sequential
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping
import tensorflow as tf
import json
import os


# GPU 设置
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


#设置图片的大小，路径，batch_size,epoch
im_height = 224
im_width = 224
batch_size = 80
epochs = 5


"""
加载本地数据集
"""
image_path = r"C:\Users\hp\Desktop\dogvscat"
train_dir = os.path.join(image_path, 'training_set/')
test_dir = os.path.join(image_path, 'test_set/')

train_dir_cats = os.path.join(train_dir, 'cats')
train_dir_dogs = os.path.join(train_dir, 'dogs')

test_dir_cats = os.path.join(test_dir, 'cats')
test_dir_dogs = os.path.join(test_dir, 'dogs')

num_cats_tr = len(os.listdir(train_dir_cats))
num_dogs_tr = len(os.listdir(train_dir_dogs))

num_cats_ts = len(os.listdir(test_dir_cats))
num_dogs_ts = len(os.listdir(test_dir_dogs))

total_train = num_cats_tr + num_dogs_tr
total_test = num_cats_ts + num_dogs_ts
# print(num_dogs_ts, num_cats_ts, num_dogs_tr, num_cats_tr)
# print(total_test, total_train)
"""
生成训练数据集和验证数据集。
"""
train_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255) #归一化
validation_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

"""
在为训练和验证图像定义生成器之后，flow_from_directory方法从磁盘加载图像，应用重新缩放，并将图像调整到所需的尺寸。
用binary模型分两类
"""
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(im_height, im_width),
                                                           class_mode='binary')
# output:Found 2000 images belonging to 2 classes.
val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=test_dir,
                                                              target_size=(im_height, im_width),
                                                              class_mode='binary')

"""
测试用
可视化训练图像,通过从训练生成器中提取一批图像(在本例中为32幅图像)来可视化训练图像，然后用matplotlib绘制其中五幅图像
"""
sample_test_images, _ = next(val_data_gen)
# next函数：从数据集中返回一个批处理。
# 返回值：(x_train，y_train)的形式，其中x_train是训练特征，y_train是其标签。丢弃标签，只显示训练图像。
# 该函数将图像绘制成1行5列的网格形式，图像放置在每一列中。
def plotImages(images_arr):
    history = tf.keras.models.load_model(r'G:\cat_dog.h5') 
    final_opt_a=history.predict(sample_test_images[0:20])#通过模型预测测试集
    print(final_opt_a)
    fig, axes = plt.subplots(4, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout() #会自动调整子图参数，使之填充整个图像区域
    plt.show()

plotImages(sample_test_images[:20])#结果可视化函数，在训练出cat_dog.h5模型后才可运行该函数

"""
创建模型
编译模型
训练模型
使用ImageDataGenerator(数据增强)类的fit_generator方法来训练网络。
"""
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(im_height, im_width, 3)),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()        # 查看网络的所有层

#设置callback防止过拟合
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.999):
      print("\nReached 99.9% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_test // batch_size,
    callbacks=[callbacks]
)
model.save(r'G:\cat_dog.h5')



"""
用matplotlib可视化训练结果
"""
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

