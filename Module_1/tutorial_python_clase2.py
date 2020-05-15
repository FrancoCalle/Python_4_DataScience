# -*- coding: utf-8 -*-
"""
Created on Mon May  4 20:47:15 2020

@author: Franco
"""

import copy


frutas = ['manzana', 'banana', 'naranja']
carnes = list(['puerco', 'vaca', 'pollo'])
mix = [123, ['Python', 'Stata'], 'iPhone']

# Lists
[1, 2, 3, 4, 5]
miPrimeraLista = [1, 2, 3, 4, 5]
miPrimeraLista1 = miPrimeraLista
miPrimeraLista2 = copy.copy(miPrimeraLista)
miPrimeraLista[0]
miPrimeraLista[0] = 3
miPrimeraLista[1] = 9
miPrimeraLista[1:]

miPrimeraLista
miPrimeraLista1
miPrimeraLista2

#List of lists:
miSegundaLista = [["Franco", "Yulino"], [26, 25]]
miSegundaListaNombres = miSegundaLista # This is passing by reference
miSegundaListaNombres2 = copy.deepcopy(miSegundaLista) # This is passing by reference
miSegundaLista[1][1] = 24
miSegundaLista
miSegundaListaNombres
miSegundaListaNombres2

del miSegundaLista[1]
miSegundaLista

# Combine lists

mylist = [1, 2] + [3, 4]
print(mylist)
[1, 2]*2

1 in mylist
5 in mylist
1 not in mylist
5 not in mylist

len(mylist)
min(mylist)
max(mylist)

# Methods for
mylist.append(5)
mylist

mylist.append([6,7])
mylist
mylist.remove([6,7])
mylist
mylist.extend([6,7])
mylist
mylist.insert(1,10) # insert element to a list at a location other than the end
mylist

mylist.sort()
mylist # sort list from lower to higher
mylist.index(2) # return the index of the first ocurrence of the given value in a list


# Tuples:
myTuple = 2,3,4
list(myTuple)

type(myTuple)
myTuple + (5, )

tuple([1,2])



# Dictionaries
miDict = {}

miDict = {'a':'perro', 'b':'gato'}
miDict['a']
miDict['a'] = 'canario'
miDict

miDict['c'] = 'conejo'

#Access keys within dictionary:
'a' in miDict
'canario' in miDict

#del miDict
list(miDict)
miDict.keys()
miDict.values()
miDict.items()

miDict2 = miDict.copy()



# def miPrimeraSuma(primerDigito, segundoDigito):
#
#     total = primerDigito + segundoDigito
#
#     return total
#
#
# totalSuma = miPrimeraSuma(4,7)
#
#
# def miOperacion(primerDigito, segundoDigito, tercerDigito):
#
#     total = (primerDigito + segundoDigito)/tercerDigito
#
#     return total
#
# totalOperacion = miOperacion(4,8,2)
#
#
# def miFiltro(basededatos):
#
#     dataTrabajadores = pd.DataFrame(basededatos).transpose()
#     dataTrabajadoresFinal = dataTrabajadores.loc[dataTrabajadores['Edad']>19,["Edad","Direccion"]]
#
#     return dataTrabajadoresFinal
#
#
# miFiltro(basededatos=caracteristicasTrabajadores)
#
#
# # Iterables
# for ii in range(1,10):
#     print(ii)
#
#
# (float(99)+100)
#
# a=(float(99)+100)
#
# print(a)
#  b=50
#
#  c=float(b)
#  print(c)
#
# x=9
#
# if x==9:
#     print('false')
#
# if x!=9:
#     print('true')
