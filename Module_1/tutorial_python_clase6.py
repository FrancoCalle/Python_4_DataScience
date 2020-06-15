# -*- coding: utf-8 -*-
"""

Python para Economistas: Sexta Clase
Autor: Franco Calle

- Pandas Module
- Using and creating Dataframes
- Replace and rename columns
- Slicing Dataframes
- Merge, Append
- Import, Export

"""

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt


#Creating series:
s = pd.Series(np.random.randn(100))
s1 = pd.Series(np.random.randint(1, 100, 100))
s2 = pd.Series(np.random.normal(0,1,100))
s3 = pd.Series(np.random.uniform(0,1,100))
s4 = pd.Series(np.random.binomial(1,.3,100))

#Creating Dataframes:
df = pd.DataFrame({"Random":s, "RandInt":s1, "Normal":s2, "Uniform":s3, "Binomial":s4})

print(df.head())

# Llamar una columna:
df['Uniform']

# Convocar multiples columnas:
df[['Uniform', 'Normal', 'Binomial']]

# Insertamos lista con todos los nombres de columna que queremos utilizar
columnList = ['Uniform', 'Normal', 'Binomial']
df2 = df[columnList]


# Reemplazando y renombrando columnas
df.columns

df = df.rename(columns={'Random': 'Random Variable', 'RandInt': 'Random Integer'})

df.columns


# Slicing:

# Si queremos solo las filas que tienen valores para RandInt mayor a 30
df3 = df.loc[df['Random Integer'] > 30, :]
print('Nuestra base inicial tiene',df.shape[0], 'observaciones y la base final tiene', df3.shape[0])

# Si queremos agregar mas condicionales
df4 = df.loc[(df['Random Integer'] > 30) & (df['Binomial'] == 1), :]
print('Nuestra base inicial tiene',df.shape[0], 'observaciones y la base final tiene', df4.shape[0])

# Si queremos las condiciones anteriores pero solo necesitamos una variable especifica:
df5 = df.loc[(df['Random Integer'] > 30) & (df['Binomial'] == 1), ['Normal', 'Uniform']]

#Merging:
dfNew = pd.DataFrame({'Integers': range(100), 'Name': ['Worker Number ' + str(ii) for ii in range(100)]})

df['Random Integer'].value_counts()

dfMerged1 = pd.merge(df,dfNew, left_on='Random Integer', right_on = 'Integers', how= 'inner', indicator=True)
dfMerged2 = pd.merge(df,dfNew, left_on='Random Integer', right_on = 'Integers', how= 'right', indicator=True)
dfMerged3 = pd.merge(df,dfNew, left_on='Random Integer', right_on = 'Integers', how= 'left', indicator=True)

dfMerged3['_merge'].value_counts()

# Drop a variable:
dfMerged3.drop(columns=['_merge'], inplace = True)

dfMergedCopy = dfMerged3.copy()

# Append a dataframe
dfAppend = dfMerged3.append(dfMergedCopy)

# Query un Dataframe:

print(dfAppend.columns)

q = ('(Binomial == %s) ' '& (Normal <= %s) ' '& (Uniform >= %s)') % (1,1,.5)

print(q)

reportCases = dfAppend.query(q)
print(reportCases)

# Cargar data:
