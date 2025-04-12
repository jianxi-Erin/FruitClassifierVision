# 如没有数据集,运行此文件下载Fruits-360 数据集并解压
import os
import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlretrieve
from zipfile import ZipFile

# 数据集下载和解压
dataset_url = "https://github.com/Horea94/Fruit-Images-Dataset/archive/refs/heads/master.zip"
dataset_path = "fruits.zip"
if not os.path.exists(dataset_path):
    urlretrieve(dataset_url, dataset_path)
    with ZipFile(dataset_path, 'r') as zip_ref:
        zip_ref.extractall(".")
    os.rename("Fruit-Images-Dataset-master", "fruits")