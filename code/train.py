from code.model import SimpleCNNModel, VGG16CNNModel, VGG16FinetuningModel
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import pandas as pd

def train():

    batchSize = 48
    imgSize = 128

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        '../data/trainImg',
        target_size=(imgSize, imgSize),
        batch_size=batchSize, shuffle=True,
        classes=["pos", "neg"], class_mode="binary") # categorical返回one-hot的类别，binary返回单值

    val_generator = test_datagen.flow_from_directory(
        '../data/trainImg2',
        target_size=(imgSize, imgSize),
        batch_size=batchSize,
        classes=["pos", "neg"], class_mode="binary")

    #CNN = VGG16CNNModel(load=False)
    CNN = VGG16FinetuningModel(load=True)
    #CNN = SimpleCNNModel()
    CNN.trainModel(generator=train_generator , validation_generator=val_generator, batchSize=batchSize)
    CNN.save()

    return CNN

def predict(CNN, load = False):

    # test_datagen = ImageDataGenerator(rescale=1. / 255)
    # test_generator = test_datagen.flow_from_directory(
    #     '../data/test/test/',
    #     target_size=(128, 128),
    #     batch_size=48, shuffle=False)

    if load:
        model = VGG16CNNModel(load=True)
    else:
        model = CNN
    ansArr = model.inference()
    print(ansArr.shape)

if __name__ == "__main__":

    CNN = train()
    #CNN = None
    predict(CNN, load=False)



