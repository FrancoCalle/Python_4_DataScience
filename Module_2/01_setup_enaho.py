import pandas as pd
import numpy as np
import os
path = os.getcwd()


#year = [2013, 2014, 2015, 2016, 2017]

df_path = "/home/franco/Documents/Data/Enaho/2017/603-Modulo03/enaho01a-2017-300.dta"
df = None
with open(df_path, 'rb') as f:
    df = pd.read_stata(f)

df = df[['nconglome','hogar', 'vivienda', 'codperso', 'ubigeo', 'dominio', 'estrato',
            'p300a', 'p301a', 'p304a', 'p305', 'p207', 'p208a']]

df.head()

df['id_persona'] = df.ubigeo + df.nconglome + df.vivienda + df.hogar + df.codperso  #Create identifier for each individual
df['id_persona'] = pd.Series(df['id_persona'].map(lambda x: str(x)), index = df.index)

df['dpto'] = pd.Series([x[0:2] for x in df['ubigeo']], index = df.index)            #Using List comprehension
df['dpto'] = pd.Series(df['ubigeo'].map(lambda x: str(x[0:2])), index = df.index)        #Using pandas map function

list_num = ['0'+str(x) if len(str(x)) == 1 else str(x) for x in range(26)]          #Create a list with string number for dpto
list_num = list(np.unique(df['dpto']))


list_dpto = ["Amazonas", "Ancash", "Apurimac", "Arequipa",
							"Ayacucho", "Cajamarca", "Callao", "Cusco",
							"Huancavelica", "Huanuco", "Ica", "Junín",
							"La Libertad", "Lambayeque", "Lima", "Loreto",
							"Madre de Dios", "Moquegua", "Pasco", "Piura",
							"Puno", "San Martín", "Tacna", "Tumbes", "Ucayali"]

df['name_dpto'] = ''
for i, j in zip(list_num, list_dpto):
    df['name_dpto'][df['dpto'] == i] = j


np.unique(df.name_dpto)


df_path = "/home/franco/Documents/Data/Enaho/2017/603-Modulo05/enaho01a-2017-500.dta"
df2 = None
with open(df_path, 'rb') as f:
    df2 = pd.read_stata(f)


df2['id_persona'] = df.ubigeo + df.nconglome + df.vivienda + df.hogar + df.codperso  #Create identifier for each individual
df2['id_persona'] = pd.Series(df2['id_persona'].map(lambda x: str(x)), index = df.index)
df2 = df2[['id_persona', 'p203', 'p204', 'p301a', 'p207', 'p208a', 'i524a1', 'd529t', 'i530a', 'd536', 'i538a1', 'd540t', 'i541a', 'd543', 'd544t', 'ocu500']]

result = pd.merge(df2, df, left_on='id_persona', right_on='id_persona')
result = result.rename(columns = {'p301a': "niveduc", "p207_x": "sexo", "p208a": "edad", "p301a_x": "niveduc", "p300a": "lang"})
list(result)

df2.shape[0], df.shape[0], result.shape[0]

result = result[result.id_persona.duplicated()==0]                                   #Drop all duplicated observations
df2.shape[0], df.shape[0], result.shape[0]

result.dtypes


result.niveduc.value_counts()
result.niveduc.value_counts(normalize=True)


#delimit;
result['niveduc_n']	= result.niveduc.astype("category").cat.codes #Convert to numerical variable

result['niveduc_n'][(result.niveduc_n == 0) | (result.niveduc_n == 1)| (result.niveduc_n == 2)| (result.niveduc_n == 3)] = 1
result['niveduc_n'][(result.niveduc_n == 4)] = 2
result['niveduc_n'][(result.niveduc_n == 5)] = 3
result['niveduc_n'][(result.niveduc_n == 6) | (result.niveduc_n == 8)] = 4
result['niveduc_n'][(result.niveduc_n == 7) | (result.niveduc_n == 9) | (result.niveduc_n == 10)] = 5

result = result[(result['niveduc_n']!=11) &  (result['niveduc_n']!=-1)]
result['niveduc_n'].value_counts()

result['estrato_n']	= result.estrato.astype("category").cat.codes #Convert to numerical variable
result['estrato_n'][(result['estrato_n'] <= 5)] = 0
result['estrato_n'][(result['estrato_n'] == 6) | (result['estrato_n'] == 7)] = 1

result.estrato_n.value_counts()

temp = pd.get_dummies(result.niveduc_n, prefix='niv')
result = pd.concat([result, temp], axis=1)

list(result)
result['ing_ocu_pri']	 	 =result[['i524a1', 'd529t', 'i530a', 'd536']].sum(1)
result['ing_ocu_pri'] 	     =result['ing_ocu_pri']/12
result['ing_ocu_sec']		 =result[['i538a1', 'd540t', 'i541a', 'd543']].sum(1)
result['ing_ocu_sec']		 =result['ing_ocu_sec']/12
result['ing_lab']			 =result['ing_ocu_pri'] + result['ing_ocu_sec']
result['ing_lab'][result['ing_lab'] == 0.]       = np.nan
result['ing_extra']			 =result['d544t']
result['ing_extra']		     =result['ing_extra']/12
result['ing_total']			 =result['ing_lab'] + result['ing_extra']
result['ing_total_anual']	 =result['ing_total']*12


result.ocu500.value_counts()
result['ocu500_n'] = result.ocu500.astype("category").cat.codes #Convert to numerical variable
ocupado_df = result[result['ocu500_n'] == 1]
desocupado_df = result[(result['ocu500_n'] != 1) & (result['ocu500_n'] != 4)]

np.nanmean(ocupado_df['ing_lab'][result.sexo == 'hombre'])
np.nanmean(ocupado_df['ing_lab'][result.sexo == 'mujer'])

np.nanmean(ocupado_df['ing_ocu_pri'][result.sexo == 'hombre'])
np.nanmean(ocupado_df['ing_ocu_pri'][result.sexo == 'mujer'])

np.nanmean(ocupado_df['ing_ocu_sec'][result.sexo == 'hombre'])
np.nanmean(ocupado_df['ing_ocu_sec'][result.sexo == 'mujer'])

np.nanmean(ocupado_df['ing_lab'][result.sexo == 'hombre'])
np.nanmean(ocupado_df['ing_lab'][result.sexo == 'mujer'])

list(ocupado_df)

ocupado_df[['sexo', 'niveduc_n']].groupby(['sexo']).mean()

ocupado_df[['ing_ocu_pri', 'ing_ocu_sec', 'ing_lab', 'ing_extra', 'ing_total', 'ing_total_anual', 'sexo', 'niveduc_n']].groupby(['sexo']).mean()
ocupado_df[['ing_ocu_pri', 'ing_ocu_sec', 'ing_lab', 'ing_extra', 'ing_total', 'ing_total_anual', 'sexo', 'niveduc_n']].groupby(['sexo']).median()
ocupado_df[['ing_ocu_pri', 'ing_ocu_sec', 'ing_lab', 'ing_extra', 'ing_total', 'ing_total_anual', 'sexo', 'niveduc_n']].groupby(['sexo']).sum()

ocupado_df[['ing_ocu_pri', 'ing_ocu_sec', 'ing_lab', 'ing_extra', 'ing_total', 'ing_total_anual', 'sexo', 'niveduc_n']].groupby(['sexo', 'niveduc_n']).mean()
ocupado_df[['ing_ocu_pri', 'ing_ocu_sec', 'ing_lab', 'ing_extra', 'ing_total', 'ing_total_anual', 'sexo', 'niveduc_n']].groupby(['sexo', 'niveduc_n']).median()
ocupado_df[['ing_ocu_pri', 'ing_ocu_sec', 'ing_lab', 'ing_extra', 'ing_total', 'ing_total_anual', 'sexo', 'niveduc_n']].groupby(['sexo', 'niveduc_n']).sum()

ocupado_df.estrato_n.value_counts()

from scipy import stats

#Creamos dataframes para utilizarlos en el t statistic
ocu_h = ocupado_df[ocupado_df['sexo']=='hombre']
ocu_m = ocupado_df[ocupado_df['sexo']=='mujer']

ocu_c = ocupado_df[ocupado_df['lang']=='castellano']
ocu_q = ocupado_df[ocupado_df['lang']=='quechua']

ocu_r = ocupado_df[ocupado_df['estrato_n']==0]
ocu_u = ocupado_df[ocupado_df['estrato_n']==1]


stats.ttest_ind(ocu_h['ing_total'].dropna(), ocu_m['ing_total'].dropna()) #Rechazamos la hipotesis nula de que hombres y mujeres ganan igual
stats.ttest_ind(ocu_c['ing_total'].dropna(), ocu_q['ing_total'].dropna()) #Rechazamos la hipotesis nula de que hombres y mujeres ganan igual
stats.ttest_ind(ocu_u['ing_total'].dropna(), ocu_r['ing_total'].dropna()) #Rechazamos la hipotesis nula de que hombres y mujeres ganan igual


ocupado_df.to_csv("/home/franco/Documents/GitHub/Python_4_DataScience/outputs/ocupados.csv")
