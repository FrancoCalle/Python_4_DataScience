'''

Python para Economistas: Segunda Clase
Autor: Franco Calle

Temas a tocar:
--------------

- Listas
- Diccionarios
- Tuplas
- Asignacion multiple con diccionarios:

'''

import copy

# Listas:
frutas = ['manzana', 'banana', 'naranja']
carnes = list(['puerco', 'vaca', 'pollo'])
mix = [123, ['Python', 'Stata'], 'iPhone']

# Asignacion como Referencias o Valores:
[1, 2, 3, 4, 5]
miPrimeraLista = [1, 2, 3, 4, 5]
miPrimeraLista1 = miPrimeraLista # Esto asigna como referencia
miPrimeraLista2 = copy.copy(miPrimeraLista) # Esto asigna como valor
miPrimeraLista[0]
miPrimeraLista[0] = 3
miPrimeraLista[1] = 9
miPrimeraLista[1:]

miPrimeraLista
miPrimeraLista1
miPrimeraLista2

#List of lists:
miSegundaLista = [["Franco", "Yulino"], [26, 25]]
miSegundaListaNombres = miSegundaLista # Esto asigna como referencia
miSegundaListaNombres2 = copy.deepcopy(miSegundaLista) # Deepcopy asigna como valor las listas y sublistas
miSegundaLista[1][1] = 24
miSegundaLista
miSegundaListaNombres
miSegundaListaNombres2

del miSegundaLista[1]
miSegundaLista

# Combinar listas
mylist = [1, 2] + [3, 4]
print(mylist)
[1, 2]*2

# Evaluar si un objeto se encuentra en una lista
1 in mylist
5 in mylist
1 not in mylist
5 not in mylist

# Metodos en listas
len(mylist) #Tamano de lista
min(mylist) #Valor minimo en la lista
max(mylist) #Valor maximo en la lista

# Metodos para agregar y eliminar elementos de una lista:
mylist.append(5) #Agregar elemento
mylist

mylist.append([6,7])
mylist
mylist.remove([6,7]) # Remover elemento
mylist
mylist.extend([6,7]) # Extender lista con mas elementos
mylist
mylist.insert(1,10) # insert element to a list at a location other than the end
mylist

mylist.sort()
mylist # Ordenar lista de menor a mayor
mylist.index(2) # Retorna el indice del valor que solicitamos


# Tuplas:
myTuple = 2,3,4
list(myTuple)

type(myTuple)
myTuple + (5, )

tuple([1,2])



# Diccionarios
miDict = {}

miDict = {'a':'perro', 'b':'gato'}
miDict['a']
miDict['a'] = 'canario'
miDict

miDict['c'] = 'conejo'

#Acceder a llaves (keys) en los diccionarios:
'a' in miDict
'canario' in miDict

list(miDict)
miDict.keys()
miDict.values()
miDict.items()

miDict2 = miDict.copy()
