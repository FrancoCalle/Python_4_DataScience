import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt


print("1. CARGAR BASES DE DATOS...")

reportePositivos = pd.read_csv(os.path.join('Module_1', 'DATOSABIERTOS_SISCOVID.csv'), encoding='iso8859_2')
reportePositivos.columns = reportePositivos.columns.str.lower() #Pasar nombre de columnas a minuscula
reportePositivos = reportePositivos.dropna(subset=['fecha_nacimiento','departamento','provincia','distrito']) # Eliminar filas que contienen Nan en alguno de las variables

reporteFallecidos = pd.read_csv(os.path.join('Module_1', 'fallecidos_minsa_covid19.csv'), encoding='iso8859_2')
reporteFallecidos.columns = reporteFallecidos.columns.str.lower()
reporteFallecidos.shape
reporteFallecidos = reporteFallecidos.dropna(subset=['fecha_nacimiento','departamento','provincia','distrito'])

dataDistritos = pd.read_excel(os.path.join('Module_1','Información de Distritos 2020.xlsx'),sheet_name="Nivel Distrital")
dataDistritos = dataDistritos.iloc[:1873,:]

print("2. CLEAN DISTRICT DATA...")

# Eliminar tildes y articulos que podr'ian ser diferentes entre distintas BDD
def replaceCharacters(nameOld):

    nameOld=nameOld.replace('Á', 'A')
    nameOld=nameOld.replace('É', 'E')
    nameOld=nameOld.replace('Í', 'I')
    nameOld=nameOld.replace('Ó', 'O')
    nameOld=nameOld.replace('Ú', 'U')
    nameOld=nameOld.replace('Ñ', 'N')

    nameOld=nameOld.replace(' DE ', '')
    nameOld=nameOld.replace(' DEL ', '')
    nameOld=nameOld.replace(' LA ', '')
    nameOld=nameOld.replace(' LAS ', '')
    nameOld=nameOld.replace(' ', '')

    nameNew = nameOld

    return nameNew

dataDistritos['grupo_etario1'] = (dataDistritos['De 0  a 4 años\n2017'] + dataDistritos['De 5  a 9 años\n2017'] +
                                    dataDistritos['De 10 a 14 años\n2017'] + dataDistritos['De 15 a 19 años\n2017'])
dataDistritos['grupo_etario2'] = (dataDistritos['De 20 a 24 años\n2017'] + dataDistritos['De 25 a 29 años\n2017'] +
                                    dataDistritos['De 30 a 34 años\n2017'] + dataDistritos['De 35 a 39 años\n2017'] + dataDistritos['De 40 a 44 años\n2017'])
dataDistritos['grupo_etario3'] = (dataDistritos['De 45 a 49 años\n2017'] + dataDistritos['De 50 a 54 años\n2017'] +
                                    dataDistritos['De 55 a 59 años\n2017'] + dataDistritos['De 60 a 64 años\n2017'])
dataDistritos['grupo_etario4'] = (dataDistritos['De 65 a 69 años\n2017'] + dataDistritos['De 70 a 74 años\n2017'])

dataDistritos['grupo_etario5'] = (dataDistritos['De 75 a 79 años\n2017'] + dataDistritos['De 80 a 84 años\n2017'] +
                                    dataDistritos['De 85 a 89 años\n2017'] + dataDistritos['De 90 a 94 años\n2017'] + dataDistritos['De 95 a más\n2017'])

dataDistritos = dataDistritos[['UBIGEO','Departamento','Provincia','Distrito','grupo_etario1','grupo_etario2','grupo_etario3','grupo_etario4','grupo_etario5']]
dataDistritos.columns = dataDistritos.columns.str.lower()
dataDistritos['departamento'] = dataDistritos['departamento'].str.upper()
dataDistritos['provincia'] = dataDistritos['provincia'].str.upper()
dataDistritos['distrito'] = dataDistritos['distrito'].str.upper()

#Save distrito name for later:
dataDistritosName = dataDistritos[['ubigeo','departamento','provincia','distrito']].copy()

dataDistritos['departamento'] = list(map(lambda x: replaceCharacters(x), dataDistritos['departamento']))
dataDistritos['provincia'] = list(map(lambda x: replaceCharacters(x), dataDistritos['provincia']))
dataDistritos['distrito'] = list(map(lambda x: replaceCharacters(x), dataDistritos['distrito']))

dataDistritos = pd.wide_to_long(dataDistritos, stubnames="grupo_etario", i=['ubigeo','departamento','provincia','distrito'], j="agecat").reset_index().rename(columns={'grupo_etario':'nHabitantes','agecat':'grupo_etario'})

print("3. MODIFY TIMESTAMPS POSITIVOS...")

reportePositivos = reportePositivos.dropna(subset=['fecha_prueba'])

reportePositivos['timestamp_fecnac'] =  list(map(lambda x:
                                datetime.strptime(x, '%Y-%m-%d') if '-' in x else datetime.strptime(x, '%d/%m/%Y'),
                                reportePositivos['fecha_nacimiento']))

reportePositivos['timestamp_prueba'] =  list(map(lambda x:
                                datetime.strptime(x, '%Y-%m-%d') if '-' in x else datetime.strptime(x, '%d/%m/%Y'),
                                reportePositivos['fecha_prueba']))

reportePositivos['edad'] = (reportePositivos['timestamp_prueba'] - reportePositivos['timestamp_fecnac']).dt.days/365

reportePositivos = reportePositivos.loc[reportePositivos['timestamp_prueba']>datetime(2020, 1, 1),:]

reportePositivos['timestamp_day'] = reportePositivos['timestamp_prueba']

reportePositivos['d_pcr'] = reportePositivos['tipo_prueba'] == 'PCR'

reportePositivos['count'] = 1

#Armar categoria por grupo etario
reportePositivos['grupo_etario'] = np.nan
reportePositivos.loc[(reportePositivos['edad']>=-1) & (reportePositivos['edad']<=19),'grupo_etario']   = 1
reportePositivos.loc[(reportePositivos['edad']>19) & (reportePositivos['edad']<=44),'grupo_etario']  = 2
reportePositivos.loc[(reportePositivos['edad']>44) & (reportePositivos['edad']<=64),'grupo_etario']  = 3
reportePositivos.loc[(reportePositivos['edad']>64) & (reportePositivos['edad']<=74),'grupo_etario']  = 4
reportePositivos.loc[reportePositivos['edad']>74 ,'grupo_etario'] = 5

#Replace Tildes and Nhes
reportePositivos['departamento'] = list(map(lambda x: replaceCharacters(x), reportePositivos['departamento']))
reportePositivos['provincia'] = list(map(lambda x: replaceCharacters(x), reportePositivos['provincia']))
reportePositivos['distrito'] = list(map(lambda x: replaceCharacters(x), reportePositivos['distrito']))


print("4. MODIFY TIMESTAMPS FALLECIDOS...")
reporteFallecidos['timestamp_fecnac'] =  list(map(lambda x:
                                datetime.strptime(x, '%d/%m/%Y') if isinstance(x,str) else np.nan,
                                reporteFallecidos['fecha_nacimiento']))

reporteFallecidos['timestamp_fecfallece'] =  list(map(lambda x:
                                datetime.strptime(x, '%d/%m/%Y') if isinstance(x,str) else np.nan,
                                reporteFallecidos['fecha_fallecimiento']))

reporteFallecidos['edad'] = (reporteFallecidos['timestamp_fecfallece'] - reporteFallecidos['timestamp_fecnac']).dt.days/365

reporteFallecidos['timestamp_day'] = reporteFallecidos['timestamp_fecfallece']

reporteFallecidos['count'] = 1

#Armar categoria por grupo etario
reporteFallecidos['grupo_etario'] = np.nan
reporteFallecidos.loc[(reporteFallecidos['edad']>=-1) & (reporteFallecidos['edad']<=19),'grupo_etario']   = 1
reporteFallecidos.loc[(reporteFallecidos['edad']>20) & (reporteFallecidos['edad']<=44),'grupo_etario']  = 2
reporteFallecidos.loc[(reporteFallecidos['edad']>44) & (reporteFallecidos['edad']<=64),'grupo_etario']  = 3
reporteFallecidos.loc[(reporteFallecidos['edad']>64) & (reporteFallecidos['edad']<=74),'grupo_etario']  = 4
reporteFallecidos.loc[reporteFallecidos['edad']>=74 ,'grupo_etario'] = 5

#Replace Tildes and Nhes
reporteFallecidos['departamento'] = list(map(lambda x: replaceCharacters(x), reporteFallecidos['departamento']))
reporteFallecidos['provincia'] = list(map(lambda x: replaceCharacters(x), reporteFallecidos['provincia']))
reporteFallecidos['distrito'] = list(map(lambda x: replaceCharacters(x), reporteFallecidos['distrito']))

# reporteFallecidos['departamento'].value_counts()

print("5. MERGE DATASETS AT DISTRICT LEVEL (1) N HABITANTES (2) N POSITIVE CASES (3) N DESEASED...")

# Numero de Casos Positivos
positiveCasesDistrito = (reportePositivos.loc[:,['departamento','provincia','distrito','grupo_etario','count']]
                        .groupby(['departamento','provincia','distrito','grupo_etario'])
                        .sum()
                        .reset_index()
                        .rename(columns={'count':'nPositivos'}))

# Numero de Fallecidos
fallecidosDistrito = (reporteFallecidos.loc[:,['departamento','provincia','distrito','grupo_etario','count']]
                        .groupby(['departamento','provincia','distrito','grupo_etario'])
                        .sum()
                        .reset_index()
                        .rename(columns={'count':'nFallecidos'}))

# Merge aggregated statistics by districts
districtsLevelData = (dataDistritos
                        .merge(positiveCasesDistrito, on=['departamento','provincia','distrito','grupo_etario'], how='left', indicator=True).rename(columns={'_merge':'_mergePositivos'})
                        .merge(fallecidosDistrito, on=['departamento','provincia','distrito','grupo_etario'], how='left', indicator=True).rename(columns={'_merge':'_mergeFallecidos'}))

districtsLevelData = districtsLevelData.loc[districtsLevelData['_mergePositivos']=='both',:].drop(['_mergeFallecidos', '_mergePositivos'], axis=1)


districtsLevelData = districtsLevelData.sort_values('nFallecidos',ascending=False)
districtsLevelData['infectadosPCP'] = (districtsLevelData['nPositivos']/districtsLevelData['nHabitantes'])*1000
districtsLevelData['muertesPCP'] = (districtsLevelData['nFallecidos']/districtsLevelData['nHabitantes'])*1000
districtsLevelData['tasa_fatalidad'] = (districtsLevelData['nFallecidos']/districtsLevelData['nPositivos'])*100
districtsLevelData.loc[districtsLevelData['nPositivos']<=30,'tasa_fatalidad'] = np.nan

districtsLevelData.sort_values('muertesPCP',ascending=False).head(10)

districtsLevelData['ubigeo'] = districtsLevelData['ubigeo'].astype(int)
districtsLevelData = districtsLevelData.drop(['departamento', 'provincia', 'distrito'],axis=1)


#Get departamento, provincia and distrito name again:
districtsLevelData = districtsLevelData.merge(dataDistritosName, on='ubigeo', how='left')
districtsLevelData = districtsLevelData[['ubigeo', 'departamento', 'provincia', 'distrito', 'grupo_etario', 'nPositivos', 'nFallecidos','nHabitantes', 'infectadosPCP', 'muertesPCP', 'tasa_fatalidad']]
districtsLevelData = districtsLevelData.dropna(subset=['ubigeo'])

#Export Final Dataset:
# districtsLevelData.to_csv(os.path.join(pathVisualizaciones,'data','contagios_fallecidos_distritos.csv'), index=False)

#Plot Fatality Rate
varName= 'infectadosPCP'
a_plot = sns.kdeplot(districtsLevelData.loc[districtsLevelData['grupo_etario']==2, varName],color='blue',label='[19 44]')
a_plot = sns.kdeplot(districtsLevelData.loc[districtsLevelData['grupo_etario']==3, varName],color='orange',label='[45 64]')
a_plot = sns.kdeplot(districtsLevelData.loc[districtsLevelData['grupo_etario']==4, varName],color='green',label='[65 74]')
a_plot = sns.kdeplot(districtsLevelData.loc[districtsLevelData['grupo_etario']==5, varName],color='black',label='[75 100]')
a_plot.set(xlim=(0, 30))



#GET NUMBER
def getDailyCases(reporteCasos, location, age_range=(0,100)):

    #Generate query by age:
    q = ('((edad >= %s) ' '& (edad <= %s))') % age_range

    #QUERY LOCATION:
    if len(location) == 1:
        q = q + " & " + ('(departamento == \"%s\") ') % location
    elif len(location) == 2:
        q = q + " & " + ('(departamento == \"%s\") ' '& (provincia == \"%s\") ') % location
    elif len(location) == 3:
        q = q + " & " + ('(departamento == \"%s\") ' '& (provincia == \"%s\") ' '& (distrito == \"%s\")') % location
    else:
        q = q

    #QUERY AGE RANGE & LOCATION:
    reporteCasos = reporteCasos.query(q)

    #COMPUTE DAILY CASES FOR THE OBSERVATIONS FILTERED
    dailyCases = (reporteCasos
                .loc[:,'timestamp_day']
                .value_counts()
                .reset_index()
                .sort_values('index') #Sort by date
                .rename(columns={'index':'date','timestamp_day':'numberCases'}) #Rename entities
                .reset_index(drop=True))

    dailyCases = dailyCases.loc[dailyCases.numberCases>10,:] # Get only days with >10 cases

    return dailyCases


# COMPUTE MOVING AVERAGE BY WITH A BW WINDOW:
def getCovidMovingAverage(dailyCases: pd.DataFrame,
                            bw: int=3):

    covidCasesMA = []
    datesCovid   = []

    nDays = len(dailyCases)
    for ii in range(nDays):
        if (ii > bw) & (ii < (nDays - bw + 1)):
            lb = ii - bw
            ub = ii + bw
            nCases = dailyCases.iloc[lb:ub,1].mean()
            dateC = dailyCases.iloc[ii,0]
            covidCasesMA.append(nCases)
            datesCovid.append(dateC)

    return covidCasesMA, datesCovid

#COMPUTE THE MAIN

def mainPositivos():

    #Get moving average for different cases in certain regions:
    location = 'All data'

    covidCasesMA_G1, datesCovid_G1 = getCovidMovingAverage(getDailyCases(reportePositivos, location, age_range=(0,19)), bw=4)

    covidCasesMA_G2, datesCovid_G2 = getCovidMovingAverage(getDailyCases(reportePositivos, location, age_range=(19,44)), bw=4)

    covidCasesMA_G3, datesCovid_G3 = getCovidMovingAverage(getDailyCases(reportePositivos, location, age_range=(44,64)), bw=4)

    covidCasesMA_G4, datesCovid_G4 = getCovidMovingAverage(getDailyCases(reportePositivos, location, age_range=(64,74)), bw=4)

    covidCasesMA_G5, datesCovid_G5 = getCovidMovingAverage(getDailyCases(reportePositivos, location, age_range=(74,100)), bw=4)

    #Plot the figure by different Ages:
    plotG1 = plt.plot(datesCovid_G1,covidCasesMA_G1,color='red',label='[0 17]')
    plotG2 = plt.plot(datesCovid_G2,covidCasesMA_G2,color='blue',label='[18 44]')
    plotG3 = plt.plot(datesCovid_G3,covidCasesMA_G3,color='orange',label='[44 64]')
    plotG4 = plt.plot(datesCovid_G4,covidCasesMA_G4,color='green',label='[65 74]')
    plotG5 = plt.plot(datesCovid_G5,covidCasesMA_G5,color='black',label='[75 100]')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.xticks(rotation=45)


def mainFallecidos():

    #Get moving average for different cases in certain regions:
    location = "ALL SAMPLE"

    covidCasesMA_G2, datesCovid_G2 = getCovidMovingAverage(getDailyCases(reporteFallecidos, location, age_range=(0,44)))

    covidCasesMA_G3, datesCovid_G3 = getCovidMovingAverage(getDailyCases(reporteFallecidos, location, age_range=(45,64)))

    covidCasesMA_G4, datesCovid_G4 = getCovidMovingAverage(getDailyCases(reporteFallecidos, location, age_range=(65,74)))

    covidCasesMA_G5, datesCovid_G5 = getCovidMovingAverage(getDailyCases(reporteFallecidos, location, age_range=(75,100)))

    #Plot the figure by different Ages:
    # plotG1 = plt.plot(datesCovid_G1,covidCasesMA_G1,color='red',label='[0 17]')
    plotG2 = plt.plot(datesCovid_G2,covidCasesMA_G2,color='blue',label='[0 44]')
    plotG3 = plt.plot(datesCovid_G3,covidCasesMA_G3,color='orange',label='[44 64]')
    plotG4 = plt.plot(datesCovid_G4,covidCasesMA_G4,color='green',label='[65 74]')
    plotG5 = plt.plot(datesCovid_G5,covidCasesMA_G5,color='black',label='[75 100]')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.xticks(rotation=45)


def main2():

    dailyCases = getDailyCases(reportePositivos, ("LIMA","LIMA"), age_range=(0,100))
    covidCasesMA, datesCovid = getCovidMovingAverage(getDailyCases(reportePositivos, ("LIMA"), age_range=(0,100)))
    Plot Figures
    plt.bar(dailyCases['date'],dailyCases['numberCases'])
    plt.plot(datesCovid,covidCasesMA,color='red')
    plt.xticks(rotation=45)






# dailyCases = reporteCasos.loc[reporteCasos['departamento']=='LAMBAYEQUE','timestamp_prueba_int'].value_counts().sort_values()
# dailyCases = dailyCases[dailyCases>10]
# plt.bar(range(dailyCases.shape[0]),dailyCases)
# plt.plot(covidCasesMA)
#
#
# reporteCasos['departamento'].value_counts()
#
#
# ii = 3
#
# resultData = reporteCasos.merge(reporteFallecidos, on='uuid', how='left', indicator=True)
#
# reporteFallecidos.shape
#
# deseasedCases = resultData.loc[resultData['_merge']=='both']
# deseasedCasesLima = resultData.loc[(resultData['departamento_y']=='LIMA') & (resultData['_merge']=='both')]
#
# deseasedCases['tipo_prueba'].value_counts()
#
# reporteCasos.tipo_prueba.value_counts()
#
# reporteCasos.departamento.value_counts()/reporteCasos.shape[0]
#
