#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main.py
~~~~~~~

"""

# Импорт библиотек
import sys
sys.path.append('src')
import os
os.chdir(os.path.dirname(__file__))


from src.neural_networks_lib import Fully_Connected_NN
import src.supplementary_lib as spplib
from src.read_mnist_dataset import MnistDataloader


NN = Fully_Connected_NN(layers_sizes = [784, 30, 10],
                        af_list = [spplib.logistic_func],
                        af_deriv_list = [spplib.logistic_func_deriv],
                        random_state = 28)



# Создание тренировочных и тестовых массивов
input_path = 'data/MNIST Dataset'
training_images_filepath = os.path.join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = os.path.join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = os.path.join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = os.path.join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

mnist_dataloader = MnistDataloader(training_images_filepath,
                                   training_labels_filepath,
                                   test_images_filepath,
                                   test_labels_filepath)
(X_train, y_train), (X_test, y_test) = mnist_dataloader.load_data()



# Предобработка данных
"""
В датасете с сайта kaggle яркость пикселя измеряется целым числом в диапазоне от 0 до 255.
Для более эффективного обучения эти числа приводятся к дробным числам в диапазоне от 0 до 1.

Далее 1-D массив признаков|меток приводится к 2-D векторной форме.
"""

import numpy as np
X_train = X_train.astype(np.float32)/255.0
X_test = X_test.astype(np.float32)/255.0

X_train = X_train.reshape(60000,-1,1)
y_train = y_train.reshape(60000,-1,1)

X_test = X_test.reshape(10000,-1,1)
y_test = y_test.reshape(10000,-1,1)


# Обучение
NN.train(X_train = X_train, 
         y_train = y_train,
         epochs = 4,
         batch_size = 10,
         eta = 3.0,
         loss_func = spplib.MSE_halved,
         loss_func_deriv = spplib.MSE_halved_deriv,
         random_state = None)



# Тестирование
NN.test(X_test, y_test, loss_func = spplib.MSE_halved)


