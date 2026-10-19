import numpy as np
import math as mt

'''
Modelo: 
    Escolhi um modelo didático e simples. É a função de Ishigami
'''

def model_new(sample):
    a = 7.0
    b = 0.1

    x1 = sample[:, 0]
    x2 = sample[:, 1]
    x3 = sample[:, 2]

    y = np.sin(x1) + a * (np.sin(x2) ** 2) + b * np.sin(x1) * (x3 ** 4)

    return y.reshape(-1, 1)