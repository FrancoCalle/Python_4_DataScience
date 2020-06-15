# -*- coding: utf-8 -*-
"""

Python para Economistas: Quinta Clase
Autor: Franco Calle

- The flow of execution, arguments, and parameters
- Adding new functions
- Definitions and uses
- Annonimous functions Lambda
- Mapping and filtering
- Numpy module

"""


import numpy as np
import pandas as pd
import time
import random
import matplotlib.pyplot as plt



# Creando una funcion

#Ejemplo 1

def miPrimeraSuma(a,b):
    return a+b

miPrimeraFuncion(1,2)
suma = miPrimeraFuncion(1,2)
print(suma)

#Ejemplo 2

def miSegundaSuma(a=None,b=None):
    if a != None and b != None:
        return a+b
    elif a == None or b != None:
        return b
    elif a != None or b == None:
            return a
    else:
        return None

miSegundaSuma(None,None)

# Funciones anonimas usando lambdas
suma = lambda x, y: x + y
suma(1,2)

resta = lambda x, y: x - y
resta(1,2)

# Mapping a function over lists:

dataX = [3, 4 , 6, 5, 10, 15]

dataY = [5, 7 , 0, 8, 2, 4]

map(suma, dataX, dataY)

list(map(suma, dataX, dataY))
list(map(resta, dataX, dataY))

list(map(lambda x, y: x + y, dataX, dataY))
list(map(lambda x, y: x - y, dataX, dataY))

# Filtrando una list:
alphabets = ['a', 'b', 'c', 'e', 'i', 'j', 'u']

# Funcion que filtra vocales:
def filtraVocales(letra):
    vocales = ['a', 'e', 'i', 'o', 'u']
    if(letra in vocales):
        return True
    else:
        return False

filteredVowels = filter(filtraVocales, alphabets)
list(filteredVowels)

filteredVowels = list(filter(lambda x: x in ['a', 'e', 'i', 'o', 'u'], alphabets))

# Numpy module

import numpy as np

miMatrix = np.ones((5,5))
miMatrix = np.ones((5,5,5))
miMatrix = np.zeros((5,5))
miMatrix = np.random.rand(5,5)
print(miMatrix)

# Numpy Arrays has multiple methods included
myArray = np.array([[1, 2],[1, 2]])
myArray.sum(0)
myArray.sum(1)
myArray.mean(1)
myArray.max(1)
myArray.min(1)
myArray.max(1)
myArray.std(1)
myArray.transpose()


for row in miMatrix:
    print(row)

for col in miMatrix.transpose():
    print(col)

#Variable aleatoria normalmente distribuida:

xi = np.random.normal(1,.5)  # Variable aleatoria con media 1 y variancia 5
X = np.random.normal(1,.5, 10000)  # Mil ocurrencias:

Xbar = X.mean()
Sigma2 = sum((Xbar-X)**2)/X.shape[0]
Sigma = np.sqrt(Sigma2)

plt.hist(X)

#Variable aleatoria que proviene de una binomial:
X = np.random.binomial(100, 0.2, 100000)  # Variable aleatoria con media 1 y variancia 5
plt.hist(X)
(X>=30).mean() # Probabilidad de que la moneda caiga cara 30 veces de las 100


# Cual es la probabilidad que hayan dos terremotos dos dias seguidos en Peru
T = np.random.binomial(1, 0.02, 1000000)  # Variable aleatoria con media 1 y variancia 5

ii = 0
for dd in range(1,1000000):
    if T[dd] == 1 and T[dd-1] == 1:
        ii += 1

print(str(ii), 'veces en', str(round(1000000/365)), "Anhos")
