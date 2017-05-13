import numpy as np
import pandas as pd
import PIL
from PIL import Image
import os

def fileClean():

    label = pd.read_csv("../data/train_labels.csv", index_col="name")
    path = "../data/train/train/"
    posPath = "../data/trainImg/pos/"
    negPath = "../data/trainImg/neg/"
    for parent,dirnames,filenames in os.walk(path):    #三个参数：分别返回1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字
        for filename in filenames:
            id = int(filename.split(".")[0])
            print("This is %d img"  %id)
            species = label.loc[id].values[0]
            img = Image.open(path + filename)
            if species == 1:
                img.save(posPath + filename)
            else:
                img.save(negPath + filename)

if __name__ == "__main__":
    fileClean()
