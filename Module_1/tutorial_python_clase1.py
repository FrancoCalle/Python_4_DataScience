# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


#A. String Data:

"Franco Calle"
nombre = "Franco Calle"

# Operations with strings:
firstName = "Franco"
lastName  = "Calle"

fullName = firstName + ' ' + lastName
print(fullName)


# Asking for inputs to the user:
beautiful_number  = input("Tell me a beautiful number")
print(beautiful_number)


# Repetition:
print("a" * 5)

# Indexing and slicing
fullName[1:6]
fullName[0:6]
fullName[0:6][-1]
fullName[:]

# Basic functions:
len(firstName)

# Metodos
str = 'I am learning Python'

# Split strings
str.split(' ')
str.split(' ', maxsplit=1)

# Create string from a list:
proteinasList = ['Carne', 'Pollo', 'Cerdo', 'Pescado']
proteinas = ', '.join(proteinasList)

# Mayusculas y minusculas:
proteinas.upper()
proteinas.lower()

# Formating strings:

numbers = '%s, %s' % ('one', 'two')
numbers = '%i, %i' % (1, 2)
numbers = '%f, %f' % (1, 2)
'{}'.format(2)

'{1} {0}'.format(1,2)


#B. Numeric Data:

3
2+2
4.5
int(4.5)
int(4.6)
print('%s %s' % (round(4.51), int(4.51)))

float(3)

#Operators:

1+2
2-1
2*3
4/2
4**2
4%3
5//2

#Some functions:

abs(-2)
round(3.1)
divmod(9,4) # First entry is quotient, second is residual
pow(4,2)

#Load package:
import math

math.pi
math.e
math.sqrt(4)
math.exp(2)
math.log(4)
math.log(math.e)
