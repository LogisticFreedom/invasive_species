from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Input, Flatten, Dense, Activation, Dropout
from keras.models import Model
from keras.models import load_model
import numpy as np
import os
import PIL
from PIL import Image
from keras import optimizers
import pandas as pd

# 模型基类
class BaseModel():

    def __init__(self, load = False): # 选择是否加载已有模型文件或构建模型训练
        if load:
            self.model = load_model("../data/vgg.h5")
        else:
            self.model = self.buildModel()
            self.model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9), metrics=['accuracy'])
        self.model.summary()

    def buildModel(self): # 子类实现具体模型结构
        pass

    def trainModel(self, generator, validation_generator, batchSize): # 训练模型

        self.model.fit_generator(generator, steps_per_epoch=2000//batchSize, epochs=17,
                                 validation_data=validation_generator,
                                 validation_steps=800//batchSize,
                                        verbose=1)

    def inference(self): # 预测

        ansList = []
        path = "../data/test/test/"
        submit = pd.read_csv("../data/sample_submission.csv", index_col = "name")

        for parent, dirnames, filenames in os.walk(path):  # 三个参数：分别返回1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字
            for index, filename in enumerate(filenames):
                id = int(filename.split(".")[0])
                print("%d image predicting...." %id)

                img = np.array(Image.open(path + filename).resize((128, 128)))
                img = img/255.0 # 归一化

                img = img.reshape(1, 128, 128, 3) # 规范维度
                ans = self.model.predict(img)
                submit.loc[id]["invasive"] = ans[0, 0]
                ansList.append(ans)
        res = np.array(ansList).transpose([2, 0, 1])[0]

        submit.to_csv("../result/result1.csv")
        print("Result has benn generated!")
        return res

    def save(self):
        pass


class VGG16CNNModel(BaseModel):

    def buildModel(self):

        # 获取预训练的卷基层
        model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
        model_vgg16_conv.summary()

        # 输入层，尺寸大小为128*128*3
        input = Input(shape=(128, 128, 3),name = 'image_input')

        # 预训练的卷基层
        output_vgg16_conv = model_vgg16_conv(input)

        # 加入的全连接层
        x = Flatten(name='flatten')(output_vgg16_conv)
        x = Dense(256, name="fc1")(x)
        x = Dropout(0.5)(x)
        x = Dense(1, activation='sigmoid')(x)

        # 建模
        my_model = Model(input=input, output=x)

        # 冻结前面的卷基层
        output_vgg16_conv.trainabel = False

        return my_model

    def save(self):
        self.model.save('../data/vgg.h5')

class VGG16FinetuningModel(VGG16CNNModel):

    def trainModel(self, generator, validation_generator, batchSize): # 训练模型

        # 冻结前25层
        for layer in self.model.layers[:25]:
            layer.trainable = False

        self.model.fit_generator(generator, steps_per_epoch=2000//batchSize, epochs=18,
                                 validation_data=validation_generator,
                                 validation_steps=800//batchSize,
                                        verbose=1)

    def save(self):

        self.model.save('../data/vgg_fine_tune_convblock.h5')


class SimpleCNNModel(BaseModel):

    def buildModel(self):

        input = Input(shape=(128, 128, 3), name='image_input')

        o1 = Convolution2D(128, 3, 3, border_mode='same', input_shape=(200, 200, 3), name="conv1")(input)
        o1 = MaxPooling2D(2, name="pool1")(o1)
        o1 = Activation('relu')(o1)

        o2 = Convolution2D(64, 3, 3, border_mode='same', name="conv2")(o1)
        o2 = MaxPooling2D(2, name="pool2")(o2)
        o2 = Activation('relu')(o2)

        o3 = Convolution2D(32, 3, 3, border_mode='same', name="conv3")(o2)
        o3 = MaxPooling2D(2, name="pool3")(o3)
        o3 = Activation('relu')(o3)

        o4 = Convolution2D(16, 3, 3, border_mode='same', name="conv4")(o3)
        o4 = MaxPooling2D(2, name="pool4")(o4)
        o4 = Activation('relu')(o4)

        x = Flatten(name="flatten")(o4)

        x = Dense(256, activation='relu', name='fc1')(x)
        x = Dense(128, activation='relu', name='fc2')(x)
        x = Dense(2, activation='softmax', name='predictions')(x)

        my_model = Model(input=input, output=x)

        my_model.summary()

        return my_model



