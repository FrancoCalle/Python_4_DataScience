'''

Python para Economistas: Primera Clase
Autor: Franco Calle

Temas a tocar:
--------------

- Values, variable names, and keywords
- Operators, operands, expressions, the order of operations, string operations
- Asking for inputs to the user
- Mnemonic variable names

'''

#1. String Data and Variable names:

"Franco Calle"
nombre = "Franco Calle"

#2. Operaciones con strings:
firstName = "Franco"
lastName  = "Calle"

fullName = firstName + ' ' + lastName
print(fullName)

#3. Numeros y operaciones con numeros:
3
1+2
2-1
2*3
4/2
4**2
4%3
5//2
4.5 # Float:
True
False

#4. Tipos de objetos:
type(3)
type(3.)
type("3.")
type(True)

#5. Cambio de tipo de elementos
int(4.5)
int(True)
int(False)
float(True)
float(3)

#6. Redondear Numeros:
round(4.5)
round(4.51)
print('%s %s' % (round(4.51), int(4.51)))


#7. Otras operaciones:
abs(-2)
round(3.1)
divmod(9,4) # First entry is quotient, second is residual
pow(4,2)


#8. Pedir insumos al usuario:
beautiful_number  = input("Tell me a beautiful number")
print(beautiful_number)


#9. Repeticiones:
print("a" * 5)

#10. Indexing y Slicing
fullName[0]
fullName[1:6]
fullName[0:6]
fullName[0:6][-1]
fullName[:]

#11. Metodos con strings
str = 'I am learning Python'

#Separar Strings
str.split(' ')
str.split(' ', maxsplit=1)

#Crear strings con una lista:
proteinasList = ['Carne', 'Pollo', 'Cerdo', 'Pescado']
proteinas = ', '.join(proteinasList)
print(proteinas)

#Mayusculas y minusculas:
proteinas.upper()
proteinas.lower()

#Formateando strings:

numbers = '%s, %s' % ('one', 'two')
numbers = '%i, %i' % (1, 2)
numbers = '%f, %f' % (1, 2)
'{}'.format(2)
'{1} {0}'.format(1,2)


#12. Mas operaciones:
import math

pi = math.pi
eps = math.e
sq4 = math.sqrt(4)
math.exp(2)
math.log(4)
math.log(math.e)
