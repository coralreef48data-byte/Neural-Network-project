#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
supplementary_lib.py
~~~~~~~~~~~~~~~~~~~~

Бибилиотека вспомогательных/элементарных функций
"""

import numpy as np




"""
Функции активации
"""

def logistic_func(x):
    """Логистическая функция"""
    return 1.0/(1.0+np.exp(-x))


def logistic_func_deriv(x):
    """Производная логистической функции"""
    elm = np.exp(-x)
    return elm/np.square(1+elm)


def logistic_func_tune(x, a=1.0, L=1.0):
    """Настраиваемая логистическая функция"""
    return L/(1.0+np.exp(-a*x))


def logistic_func_tune_deriv(x, a=1.0, L=1.0):
    """Производная настраиваемой логистической функции"""
    elm = np.exp(-a*x)
    return L*a*elm/np.square(1+elm)


def tanh(x):
    """Гиперболический тангенс"""
    return (1-np.exp(-2*x))/(1+np.exp(2*x))


def tanh_deriv(x):
    """Производная гиперболического тангенса"""
    return 1 - np.square((1-np.exp(-2*x))/(1+np.exp(-2*x)))


@np.vectorize
def ReLU(x):
    """Rectified Linear Unit"""
    if x < 0:
        return 0
    else:
        return x


@np.vectorize
def ReLU_deriv(x):
    """Derivative of Rectified Linear Unit"""
    if x < 0:
        return 0
    else:
        return 1




"""
Функции отклонения/потерь
"""


def MSE_halved(a: np.ndarray, y: np.ndarray):
    """
    Среднее квадратов отклонений.
    1/2 введена для более удобного вида производной.
    
    На вход подаются по одному вектору каждого типа, либо массив таких векторов.
    """
    if a.shape != y.shape:
        raise ValueError('Размеры массивов не совпадают')
        
    if a.ndim == 2:
        return (np.square(a-y)).sum()*0.5
    elif a.ndim == 3:
        return (np.square(a-y)).sum()*0.5/a.shape[0]
    else:
        raise ValueError('Некорректная размерность данных')


def MSE_halved_deriv(a: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Производная среднего квадратов отклонений"""
    if a.shape != y.shape:
        raise ValueError('Размерности массивов не совпадают')

    return a-y


def cross_entropy(a: np.ndarray, y: np.ndarray):
    """Кросс-энтропия"""
    return -1*np.nan_to_num(y*np.log(a) + (1-y)*np.log(1-a)).sum()
    
    
def cross_entropy_deriv(a: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Производная кросс-энтропии"""
    return -1*(y-a)/(a*(1-a))
    
    
    
    

if __name__ == '__main__':
    pass
