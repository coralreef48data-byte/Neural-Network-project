#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
neural_networks_lib.py
~~~~~~~~~~~~~~~~~~~~~~

Библиотека моих нейронных сетей.
"""

import numpy as np
from tqdm import tqdm
import csv



class Fully_Connected_NN():
    """
    Полносвязная сеть прямого распространения.
    Supervised learning.
    Задача классификации.
    При обучении используются SGD, backpropagation.
    """
    
    layers_sizes: list # список с числом нейронов каждого слоя
    lrs_amount: int # количество слоев в модели
    weights: list   # список, содержащий веса нейронов послойно
    biases: list    # список, содержащий смещения нейронов послойно
    epoch_count: int # суммарное число обучающих эпох
    af_list: list
    af_deriv_list: list
    name: str           # имя сети
    description: str    # описание сети
    
    def __init__(self, layers_sizes: list, af_list: list, af_deriv_list: list,
                 random_state = None, name: str = '-', description: str = '-') -> None:
        """
        Параметры
        ---------
        layers_sizes : list
            Список, содержащий количество нейронов в каждом слое.
            Длина данного списка определяет число слоев (с учетом входного слоя).
            layers_sizes[0] -- количество входных нейронов
            
        af_list : list
            Список функций активаций.
            Подразумевается, что у каждого слоя может быть своя функция активации.
            
        af_deriv_list : list
            Первые производные соответсвующих функций активаций.
            
        random_state : TYPE, optional
            Параметр для управления случайной генерацией начальных весов и смещений.
        """
        
        if len(layers_sizes) < 2:
            raise ValueError('Недостаточное количество слоев')
        
        self.layers_sizes = layers_sizes
        self.lrs_amount = len(layers_sizes)
        
        # Генерация параметров
        rnd = np.random.RandomState(random_state)
        self.weights = list(map(rnd.randn, layers_sizes[1:], layers_sizes[:-1]))
        self.biases = list(map(rnd.randn, layers_sizes[1:],
                                          [1]*(len(layers_sizes)-1)))
        
        self.epoch_count = 0
        
        self.name = name
        self.description = description
        
        # Реализация списка функций активаций.
        # Если переданный список неполон, то пропуски заполняются
        # последней заданной функцией активации.
        len_af = len(af_list)
        if self.lrs_amount-1 == len_af:
            self.af_list = af_list
        else:
            # Случай неполного списка
            self.af_list = [None]*(self.lrs_amount-1)
            self.af_list[:len_af] = af_list
            self.af_list[len_af:] = [af_list[-1]]*(self.lrs_amount-len_af-1)
        
        # Аналогично для производных функций активации.
        # Сами производные должны соответствовать изначальным функциям.
        len_afd = len(af_deriv_list)
        if self.lrs_amount-1 == len_afd:
            self.af_deriv_list = af_deriv_list
        else:
            # Случай неполного списка
            self.af_deriv_list = [None]*(self.lrs_amount-1)
            self.af_deriv_list[:len_afd] = af_deriv_list
            self.af_deriv_list[len_afd:] = [af_deriv_list[-1]]*(self.lrs_amount-len_afd-1)


    def summary(self):
        """
        Выводит сводку по нейронной сети.
        """
        print(f"Class: {self.__class__}")
        print(f"Name: {self.name}")
        print(f"Description: {self.description}")
        print(f"Amount of layers: {self.lrs_amount}")
        print(f"Amount of neurons in each layer: {self.layers_sizes}")
        print("Shape of weights matrices:")
        for w in self.weights:
            print(w.shape)
        print(f"Epochs passed: {self.epoch_count}")
        
        
    def train(self,
              X_train: np.ndarray,      # Массив вектор-признаков
              y_train: np.ndarray,      # Соответсвующие им метки
              epochs: int,              # Количество эпох для обучения
              batch_size: int,          # Размер одного батча
              eta: float,               # Скорость обучения
              loss_func: callable,
              loss_func_deriv: callable,
              random_state = None,      # Параметр разбиения обучающих данных
              ) -> None:
        """
        Модуль обучения нейронной сети.
        Отображение расчетов реализовано с помощью tqdm.
        """

        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError('Размеры X_train и y_train не совпадают')

        # Расчет эпох и отображение прогресса
        epoch_bar = tqdm(range(epochs))
        for epoch in epoch_bar:
            epoch_bar.set_description("Processing epochs")
            
            self.epoch_count += 1
            
            """
            Тренировочные данные перемешиваются, а затем разбиваются на батчи.
            Для этого создается вспомогательный массив индексов.
            Эти индексы перемешиваются а потом разбиваются. Тренировочные признаки
            и метки берутся по индексам, попавшим в данный батч.
            
            После этого используется градиентный спуск.
            """
            
            if random_state != None:
                rnd = np.random.RandomState(random_state)
            else:
                rnd = np.random.RandomState()
            ind_shf = np.arange(0, X_train.shape[0])
            rnd.shuffle(ind_shf)
            ind_batches = [ind_shf[m:m+batch_size] for m in range(0, X_train.shape[0], batch_size)]
            del rnd

            # Прогонка батчей
            batches_bar = tqdm(ind_batches)
            for i_b in batches_bar:
                batches_bar.set_description("Processing batches")
                
                X_batch = X_train[i_b]
                y_batch = y_train[i_b]
                w_upd, b_upd = self.Gradient_Descent(X_batch, y_batch, eta,
                                                     loss_func, loss_func_deriv)
                
                # Обновление весов и смещений
                for l in range(self.lrs_amount-1):
                    self.weights[l] += w_upd[l]
                    self.biases[l] += b_upd[l]
    
    
    def Gradient_Descent(self, X_batch: np.ndarray, y_batch: np.ndarray, eta: float,
                         loss_func: callable, loss_func_deriv: callable) -> tuple:
        """
        Градиентный спуск.
        ---
        
        Стохастическая составляющая реализована в методе self.train
        Для вычисления производных используется backpropagation.
        
        Возвращает кортеж из двух списков.
        Первый список: изменение весов.
        Второй список: изменение смещений.
        """
        
        wb_len = self.lrs_amount-1
        w_upd = list(map(np.zeros, zip(self.layers_sizes[1:], self.layers_sizes[:-1])))
        b_upd = list(map(np.zeros, zip(self.layers_sizes[1:], [1]*wb_len)))

        # для каждого вектора и метки из батча производим расчет 
        for X, y in zip(X_batch, y_batch):
            
            # задание пустых списков
            z = [None]*(wb_len)
            a = [None]*(wb_len)
            
            # прямая прогонка
            for l in range(wb_len):
                if l == 0:
                    z[0] = np.matmul(self.weights[0], X) + self.biases[0]
                else:
                    z[l] = np.matmul(self.weights[l], a[l-1]) + self.biases[l]
                a[l] = self.af_list[l](z[l])   
            
            # обратная прогонка
            delta = loss_func_deriv(a[-1], y)*self.af_deriv_list[-1](z[-1])
            w_upd[-1] += np.matmul(delta, a[-2].T)
            b_upd[-1] += delta
            
            if wb_len >= 2:
                for l in range(wb_len-2,-1,-1):
                    delta = self.af_deriv_list[l](z[l])*np.matmul(self.weights[l+1].T, delta)
                    b_upd[l] += delta
                    if l == 0:
                        w_upd[0] += np.matmul(delta, X.T)
                    else:
                        w_upd[l] += np.matmul(delta, a[l-1].T)

        # учет скорости обучения и размера батча
        coeff = -eta/X_batch.shape[0]
        w_upd = [coeff*w for w in w_upd]
        b_upd = [coeff*b for b in b_upd]
        return (w_upd, b_upd)


    def soft_max(self, vector: np.ndarray, beta: float = 1) -> np.ndarray:
        # beta -- обратная температура
        
        vector = np.exp(beta*vector)
        return vector/vector.sum()
    
    
    def arg_max(self, vector: np.ndarray) -> np.ndarray:
        v = np.zeros(vector.shape)
        v[np.argmax(vector)][0] = 1.0
        return v
        
    
    def forward(self, vector: np.ndarray, post_func = None) -> np.ndarray:
        """
        Прогонка признак-вектора через нейронную сеть.

        В конце можно применть дополнительную обработку.
        """

        for w_lr, b_lr, activ_f in zip(self.weights, self.biases,
                                       self.af_list):
            vector = np.matmul(w_lr, vector) + b_lr
            vector = activ_f(vector)
            
        if post_func != None:
            return post_func(vector)
        else:
            return vector

        
    def test(self,
             X_test: np.ndarray,    # массив признак-векторов
             y_test: np.ndarray,    # массив меток-векторов
             loss_func: callable    # функция потерь
             ):
        """
        Тестирование нейронной сети
        """
        
        if X_test.shape[0] != y_test.shape[0]:
            raise ValueError('Размерности X_test и y_test не совпадают')
        
        loss_value = 0.0
        accuracy = 0
        
        test_bar = tqdm(range(y_test.shape[0]))
        for i in test_bar:
            test_bar.set_description("Processing test")
            
            a = self.forward(X_test[i])
            loss_value += loss_func(a, y_test[i])
            if np.argmax(a) == np.argmax(y_test[i]):
                accuracy += 1
        
        loss_value /= y_test.shape[0]

        print('')
        print(f"Точность при тестировании: {accuracy} / {y_test.shape[0]}")
        print(f"Значение функции потерь при тестировании: {loss_value}")
        
    
    def save(self, filepath: str,) -> None:
        """
        Сохранение параметров сети в файл формата csv.
        
        строка 1: l1, l2, ... -- количество нейронов в каждом слое
        строка 2: EOL -- метка разделитель
        
        Далее идут строки каждой матрицы весов в соответствии с их порядком.
        (одна строка в file.csv = одна строка матрицы)
        После этого идет метка разделитель EOW.
        
        Далее идут значения смещений каждого уровня в соответсвии с их порядком.
        (одна строка в file.csv = смещения нейронов одного слоя)
        После этого идет метка разделитель EOB, указывающая на окончание считывания.
        Все, что идет за ней, при считывании будет игнорироваться.
        """

        with open(filepath, 'w', encoding='utf-8', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(self.layers_sizes)
            writer.writerow(['EOL'])
            for w in self.weights:
                writer.writerows(w)
            writer.writerow(['EOW'])
            for b in self.biases:
                writer.writerow(b.T[0])
            writer.writerow(['EOB'])
        
        
    def load(self, filepath: str) -> None:
        """
        Загрузка параметров сети из файла формата csv.
        """
        with open(filepath, 'r', encoding='utf-8', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            self.layers_sizes = list(map(int, next(reader))) 
            self.lrs_amount = len(self.layers_sizes)
            if self.lrs_amount < 2:
                raise ValueError('Недостаточное количество слоев')
            
            if next(reader)[0] == 'EOL':
                pass
            else:
                raise Exception("Ошибка считывания CSV файла: отсутствует метка 'EOL'")
                
            self.weights = list(map(np.empty, zip(self.layers_sizes[1:], self.layers_sizes[:-1])))
            for l in range(self.lrs_amount-1):
                for r in range(self.weights[l].shape[0]):
                    self.weights[l][r] = np.array(next(reader), np.float64)
                    
            if next(reader)[0] == 'EOW':
                pass
            else:
                raise Exception("Ошибка считывания CSV файла: отсутствует метка 'EOW'")
            
            self.biases = list(map(np.zeros, zip(self.layers_sizes[1:], [1]*(self.lrs_amount-1))))
            for l in range(self.lrs_amount-1):
                self.biases[l] = np.array(next(reader), np.float64).reshape(-1,1)
                
            if next(reader)[0] == 'EOB':
                print("CSV файл был успешно считан.")
            else:
                print("CSV файл был считан, однако отсутствует метка 'EOB'. Возможны ошибки.")


if __name__ == '__main__':
    pass
