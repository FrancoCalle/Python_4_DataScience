# -*- coding: utf-8 -*-
"""

Python para Economistas: Tercera Clase
Autor: Franco Calle

- Boolean Expressions
- Logical operators
- Conditional, alternative, and chained conditional executions
- Nested conditionals
- Guardians: catching expressions using try and except
- Short-circuit evaluation of logical expressions

"""

#1. Expresiones Booleanas y Operadores logicos:

import timeit

#In programming you often need to know if an expression is True or False.
#When you compare two values, the expression is evaluated and Python returns the Boolean answer:

#Comparacion
print(10 > 9)
print(10 >= 9)
print(10 == 9)
print(10 <= 9)
print(10 == 10.)

#logical
print(not True)
print(1<5 or 1>3)
print(1<5 and 1>3)
print(not(1<5 and 1>3))

#Identidad
print(10 is 10.)
print(10 is not 10.)

#Membresia
print(10 in [9, 10])
print(10 not in [9, 10])

#Evaluar el tipo de elemento
print(isinstance(10, int))
print(isinstance("10", str))
print(isinstance(10., float))
print(isinstance([], list))
print(isinstance({}, dict))
print(isinstance((1,), tuple))

#Evaluar si un elemento es True o False
bool()
bool([])
bool({})
bool(False)
bool("")
bool(0)
bool(1)
bool(None)
bool("Hello")
bool(15)

#2. Conditional, alternative, and chained conditional executions

x = 3
y = 5

# Condicionales
if x < y:
    print("x es menor que y")

# Alternativos
if x < y:
    print("x es menor que y")
else:
    print("x es mayor que y")

# Condiciones en cadena
if x < y:
    print("x es menor que y")
elif x > y:
    print("x es mayor que y")
else:
    print("x e y deberian ser iguales")

#3. Condicionales anidados

# nested if-else statement
x = -10
if x < 0:
    xx = x**2
    print("Elevamos ",  x, " al cuadrado y el resultado es ", xx)
else:
    if x > 0:
        print(x, " is a positive number")
    else:
        print(x, " is 0")

# Guardians: capturando expresiones usando try y except:

try:
  print(xsdasdw)
except:
  print("An exception occurred")


# Short circuit:

if True and False:
    print('hello world')



if True and True:
    print('hello world')
