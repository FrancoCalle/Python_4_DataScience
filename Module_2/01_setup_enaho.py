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


df['dpto'] = pd.Series([x[0:2] for x in df['ubigeo']], index = df1.index)       #Using List comprehension
df['dpto'] = pd.Series(df['ubigeo'].map(lambda x: x[0:2]), index = df1.index)   #Using pandas map function

list_num = ['0'+str(x) if len(str(x)) == 1 else str(x) for x in range(26)]      #Create a list with string number for dpto
list(np.unique(df['dpto']))


list_dpto = ["Amazonas", "Ancash", "Apurimac", "Arequipa",
							"Ayacucho", "Cajamarca", "Callao", "Cusco",
							"Huancavelica", "Huanuco", "Ica", "Junín",
							"La Libertad", "Lambayeque", "Lima", "Loreto",
							"Madre de Dios", "Moquegua", "Pasco", "Piura",
							"Puno", "San Martín", "Tacna", "Tumbes", "Ucayali"]



label 		values dpto dptol

lookfor 	sexo							//Buscamos alguna variable que contenga "sexo" dentro de su descripción
lookfor 	edad							//Buscamos alguna variable que contenga "edad" dentro de su descripción

rename		p207     sexo					//Renombramos la variable "sexo"
rename		p208a    edad					//Renombramos la variable "edad"
rename 		p301a    niv_edu				//Renombramos el nivel de educación de la persona

recode p300a (1 2 3 8= 1 "Lengua Nativa y S.M.") (4 6 7 = 0 "Castellano o L.E."), gen(lengua)

label var lengua "Lengua Materna"					//Cambiamos el label de la variable para que no aparezca "recode"

gen 		mujer = 1 if sexo == 2					//Generamos una variable binaria (Forma no eficiente)
replace 	mujer = 0 if sexo == 1

drop 		mujer

gen 		mujer = (sexo==2) if !missing(sexo) 	//Generamos la misma variable de manera eficiente

label 		define sexol 	2 "Mujer" 1 "Hombre"	//Construimos etiquetas a los valores

label 		values sexo sexol						//Colocamos las etiquetas a los valores de Sexo

											//Con Merge combinamos la base de datos de Educación con la de empleo
merge 1:1 ubigeo conglome vivienda hogar codperso using "$dir1/Sesion 2/Base s2/enaho01a-2014-500.dta", keepusing(estrato p203 p204 p301a p207 p208a i524a1 d529t i530a d536 i538a1 d540t i541a d543 d544t ocu500)

count if _merge==1 & edad<14
drop  if _merge==1
drop _merge

#delimit;															//Creamos una nueva variable para los niveles de educación
recode niv_edu 	(1/4=1 		"primaria")
				(5=2 		"secundaria incomp")
				(6=3 		"secundaria comp")
				(7 9=4 		"superior incomp")
				(8 10 11=5 	"superior comp")
				if ocu500==1 & !missing(niv_edu),
				gen(niv_edu_g);
#delimit cr

recode estrato (7 8 = 1 "Rural") (1/6=0 "Urbano"), gen(rural) 		//Generamos una variable que identifique a las personas que viven en áreas rurales o urbanas

label var rural "Estrato Geográfico"								//Cambiamos el label de la variable para que no aparezca "recode"

gen edu_prim			=(niv_edu==1 | niv_edu==2 | niv_edu==3 | niv_edu==4) 	if !missing(niv_edu)
gen edu_sec_incomp		=(niv_edu==5) 											if !missing(niv_edu)
gen edu_sec_comp		=(niv_edu==6) 											if !missing(niv_edu)
gen edu_sup_incomp		=(niv_edu==7 | niv_edu==9) 								if !missing(niv_edu)
gen edu_sup_comp		=(niv_edu==8 | niv_edu==10 | niv_edu==11) 				if !missing(niv_edu)



count if mujer== 1									//Contamos todas las mujeres
count if mujer== 0									//Contamoos todos los hombres
count if mujer==1 & niv_edu==6						//Contamos todas las mujeres que tienen secundaria completa
count if niv_edu == .								//Contamos cuantos no reportan nivel de educación

list mujer dpto niv_edu p300a edad if niv_edu==.	//Permite listar las observaciones que cumplen con niv_edu == .

bys sexo: sum edad									//Utilizamos el comando bys para realizar una orden en función de las categorías
bys sexo: sum niv_edu

numlabel, add
tabulate niv_edu_g sexo, m 							//Tabulamos con dos entradas la variable educación y sexo

table niv_edu_g, contents(n edad mean edad sd edad min edad max edad) center row col 			// Creamos un cuadro de estadísticos descriptivos para la variable educación

table sexo, contents(n edad mean edad sd edad min edad max edad) center row col format(%9.2f) 	// Creamos un cuadro de estadísticos descriptivos para la variable sexo

table dpto, contents(n niv_edu_g mean niv_edu_g sd niv_edu_g min niv_edu_g max niv_edu_g) format(%9.2f) center row col // Vemos el nivel de educación promedio por departamento

egen ing_ocu_pri	 	=rowtotal(i524a1 d529t i530a d536)						if ocu500==1 & !missing(ocu500)
replace ing_ocu_pri 	=ing_ocu_pri/12
egen ing_ocu_sec		=rowtotal(i538a1 d540t i541a d543)						if ocu500==1 & !missing(ocu500)
replace ing_ocu_sec		=ing_ocu_sec/12
egen ing_lab			=rowtotal(ing_ocu_pri ing_ocu_sec)  					if ocu500==1 & !missing(ocu500)
replace ing_lab			=. 														if ing_lab==0 & ocu500==1
egen ing_extra			=rowtotal(d544t) 										if ocu500==1 & !missing(ocu500)
replace ing_extra		=ing_extra/12
egen ing_total			=rowtotal(ing_lab ing_extra)			 				if ocu500==1 & !missing(ocu500)
gen ing_total_anual		=ing_total*12

replace ing_total = . if ing_total == 0
bys sexo : sum ing_total							//Observamos brecha salarial entre hombres y mujeres

table dpto, contents(n ing_total mean ing_total sd ing_total min ing_total max ing_total) format(%9.2f) center row col // Vemos el nivel de educación promedio por departamento

table dpto sexo, contents(mean ing_total) format(%9.2f) 		// Vemos el salario promedio por departamento y sexo

table dpto rural, contents(mean ing_total) format(%9.2f) 		// Vemos el salario promedio por departamento y area geográfica

table rural sexo lengua, contents(mean ing_total) format(%9.2f) // Vemos el salario promedio según sexo y area en la que vive y lengua materna

ttest ing_total if ocu500==1, by(sexo) 							// Realizamos test de medias para brecha x sexo
ttest ing_total if ocu500==1, by(lengua)  						// Realizamos test de medias para brecha x lengua materna
ttest ing_total if ocu500==1, by(rural) 					 	// Realizamos test de medias para brecha urbano - rural

order dpto ubigeo conglome vivienda hogar codperso dominio
sort dpto

label data "Encuesta Nacional de Hogares (ENAHO) Módulos 3 y 5"	//agregamos etiqueta a la base
describe, fullnames


*Guardando la base como spreadsheet
outsheet using "lección2_guardada.xls", replace									//guardamos la base trabajada en esta lección en xls
outsheet ubigeo ing_total sexo using "lección2_guardada2.xls", replace			//guardamos sólo las variables ubigeo, ingreso y sexo en xls
outsheet using "lección2_guardada3.xls" if dpto==15, replace					//guardamos sólo las observacions de Lima
outsheet using "lección2_guardada4.xls" in 1/7, replace							//guardamos sólo las observaciones del 1 al 7

export excel ubigeo ing_total sexo using "lección2_guardada5.xls" if dpto==15, firstrow(variables) replace		//guardamos sólo las variables ubigeo, ingreso y sexo en xls para Dpto= Lima
export excel using "lección2_guardada6.xls" if dpto==15, firstrow(variables) sheet(selec_lima) replace	//guardamos sólo las observacions de Lima en la hoja selec_if de un libro
export excel using "lección2_guardada6.xls" in 1/7, firstrow(variables) sheet(selec_7)				//guardamos sólo las observaciones del 1 al 7 en la hoja selec_in de un mismo libro


save 				"Base_de_DatosFinal", replace

*Eliminamos Archivos

erase				"Base_de_DatosFinal.dta"
erase				"lección2_guardada.xls"
erase				"lección2_guardada2.xls"
erase				"lección2_guardada3.xls"
erase				"lección2_guardada4.xls"
erase				"lección2_guardada5.xls"
erase				"lección2_guardada6.xls"
