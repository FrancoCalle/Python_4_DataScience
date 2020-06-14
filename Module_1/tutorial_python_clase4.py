# -*- coding: utf-8 -*-
"""

Python para Economistas: Tercera Clase
Autor: Franco Calle

- Updating variables
- Definite loops using for
- Double, multiple and nested iteration
- While statement
- List comprehension

"""


import numpy as np
import pandas as pd
import time
import random
import matplotlib.pyplot as plt


# Actualizando variables:

ii = 0
print(ii)
ii += 1
print(ii)
ii -= 1
print(ii)


# Loops usando for:

for ii in range(10):
    print(ii)


for ii in range(1,501):
    print("Numero: " + str(ii))

# Se puede iterar a traves de listas
numberList = [1,2,3,4,5]
for ii in numberList:
    print(ii)

# Se puede iterar a traves de tuplas
numberTuple = (1,2,3,4,5)
for ii in numberTuple:
    print(ii)

# Podemos hacer update incluyendo elementos a una lista:
listaDeNumeros = []
for ii in range(1,501):
    listaDeNumeros.append("Numero: " + str(ii))

# Hacemos lo mismo con list comprehension:
listaDeNumeros = ["Numero: " + str(ii) for ii in range(1, 501)]

# Iterable mas condicionales:
nuevaListaDeNumeros = []

for ii in range(1,50):
    if ii <= 25:
        nuevaListaDeNumeros.append("Trabajador: " + str(ii))
    if ii > 25:
        nuevaListaDeNumeros.append("Trabajadora: " + str(ii))


codigoTrabajador = list(range(1,101))
codigoTrabajadorPar = []

for ii in codigoTrabajador:
    if ii%2 == 0:
        codigoTrabajadorPar.append("Trabajador: "+ str(ii))


codigoTrabajadorMultiploSeis = []
for ii in codigoTrabajador:
    if ii%2 == 0 and ii%3 == 0:
        codigoTrabajadorMultiploSeis.append("Trabajador: "+ str(ii))


codigoTrabajadorMultiplos = []
for ii in codigoTrabajador:
    if ii%2 == 0 or ii%3 == 0:
        codigoTrabajadorMultiplos.append("Trabajador: "+ str(ii))


# Nested iteration:
zeroArray = np.zeros((5,5))
for ii in range(zeroArray.shape[0]):
    for jj in range(zeroArray.shape[1]):
        zeroArray[ii,jj] = 1




# Double iteration:

workersAge  = [random.randint(18, 50) for i in range(1000)]
workersName = ["Trabajador: " + str(i) for i in range(1,1001)]
workerRandomNumber = [random.uniform(0, 1) for i in range(1000)]


for randomNumber, name in zip(workerRandomNumber, workersName):
    print(name, '|', round(randomNumber,4))


#Grupo 1: Peru [< 0.10]
#Grupo 2: Chile [0.10 - 0.30]
#Grupo 3: Argentina [0.30 - 0.60]
#Grupo 4: Colombia  [0.60 - 1]

groupPeru = []
groupChile = []
groupArgentina = []
groupColombia = []

for randomNumber, name in zip(workerRandomNumber, workersName):
    if randomNumber < 0.10:
        groupPeru.append(name)
    elif randomNumber >= 0.10 and randomNumber < 0.30:
        groupChile.append(name)
    elif randomNumber >= 0.30 and randomNumber < 0.60:
        groupArgentina.append(name)
    else:
        groupColombia.append(name)


# Using list comprehension and functions

def groupAsignment(seed):
    if seed < 0.10:
        groupName = "Peru"
    elif seed >= 0.10 and seed < 0.30:
        groupName = "Chile"
    elif seed >= 0.30 and seed < 0.60:
        groupName = "Argentina"
    else:
        groupName = "Colombia"

    return groupName


groupPeru = [name for seed, name in zip(workerRandomNumber, workersName) if groupAsignment(seed) == "Peru"]
groupChile = [name for seed, name in zip(workerRandomNumber, workersName) if groupAsignment(seed) == "Chile"]
groupArgentina = [name for seed, name in zip(workerRandomNumber, workersName) if groupAsignment(seed) == "Argentina"]
groupColombia = [name for seed, name in zip(workerRandomNumber, workersName) if groupAsignment(seed) == "Colombia"]


# While statements:

jj = 0
while jj <= 500:
    jj = jj + 1
    time.sleep(0.3)
    if jj%100 == 0:
        print("Estamos en el numero: " + str(jj))

# Iteration in parallel:
randomNumberList= [random.random() for i in range(1000)]
plt.hist(randomNumberList, density=True)

randomNumberList= [random.uniform(5,3) for i in range(1000)]
plt.hist(randomNumberList, density=True)


randomNumberList= [random.normalvariate(0.5, 1) for i in range(1000)]
plt.hist(randomNumberList, density=True)
randomNumberList= [random.normalvariate(0.5, 2) for i in range(1000)]
plt.hist(randomNumberList, density=True)
plt.plot()
