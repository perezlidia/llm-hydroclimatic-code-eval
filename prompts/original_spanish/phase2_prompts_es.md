# Phase 2 prompts — original Spanish text (polygon clip, ArcMap reference)

Exact prompts as sent to the models (in Spanish). An English translation is in
[`../phase2_prompts.md`](../phase2_prompts.md).

---

## T1 — Temporal merging and polygon clipping

```
Tengo 45 archivos NetCDF de precipitación mensual TerraClimate y 45 de
evapotranspiración potencial, uno por año (1981 a 2025), nombrados
TerraClimate_ppt_AAAA.nc en la carpeta "D:/2027-ARTICULOS/CAPITULO-SOFTWARE/ppt/"
y TerraClimate_pet_AAAA.nc en "D:/2027-ARTICULOS/CAPITULO-SOFTWARE/pet/".
Cada archivo contiene una variable "ppt" o "pet" respectivamente, con
dimensiones lat, lon, time (12 meses). La cobertura es global.

También tengo un shapefile llamado "Noroeste.shp" (con sus archivos .shx,
.dbf, .prj asociados) en la carpeta "D:/2027-ARTICULOS/CAPITULO-SOFTWARE/Noroeste/",
que contiene los polígonos de 4 estados de México (Baja California, Baja
California Sur, Sonora y Sinaloa), en coordenadas geográficas WGS84.

Escribe un script en Python que:
1. Cargue los 45 archivos de precipitación y los unifique en un solo
   dataset con una serie temporal continua de 1981 a 2025.
2. Cargue los 45 archivos de evapotranspiración potencial y los unifique
   de la misma forma.
3. Recorte AMBOS datasets usando el POLÍGONO REAL del shapefile
   "Noroeste.shp" (no un rectángulo/bounding box) — las celdas fuera del
   polígono deben quedar como NaN/NoData, no solo excluidas por un rango
   rectangular de lat/lon.
4. Verifique que no haya fechas duplicadas y que estén en orden cronológico
   en ambos datasets, y que ambos tengan EXACTAMENTE las mismas coordenadas
   espaciales (lat/lon) y el mismo número de pasos de tiempo tras el
   recorte — repórtalo explícitamente.
5. Imprima el número total de pasos de tiempo y las dimensiones espaciales
   resultantes (filas x columnas de la grilla recortada) de cada dataset.
6. Guarde cada resultado en un nuevo archivo NetCDF, nombrados
   "ppt_noroeste_shp_1981_2025.nc" y "pet_noroeste_shp_1981_2025.nc"
   (con el sufijo "_shp" para distinguirlos de una versión anterior
   recortada con un rectángulo simple).

Usa geopandas para leer el shapefile y rioxarray (o una librería
equivalente) para hacer el recorte por geometría exacta sobre el NetCDF.
No asumas que las coordenadas de latitud vienen en un orden particular.
```

---

## T2 — Detection and handling of missing and invalid values

```
Tengo un archivo NetCDF de precipitación mensual (variable "ppt", mm/mes,
dimensiones time, lat, lon) ya recortado a una región que incluye tierra y
celdas fuera del polígono de estudio (marcadas como NaN/NoData). El archivo
se llama "ppt_noroeste_shp_CORRUPTO_prueba_T2.nc" y está en la misma
carpeta que el script.

Las celdas fuera del polígono (NaN en TODOS los pasos de tiempo) representan
zonas fuera del área de estudio, no datos faltantes reales.

Necesito un script en Python que:
1. Identifique qué celdas corresponden al área de estudio (tienen datos
   válidos en al menos algún momento de la serie) y cuáles están fuera del
   polígono (NaN en toda la serie).
2. Detecte, SOLO dentro de las celdas del área de estudio, valores que sean:
   a) NaN faltante inesperado
   b) Físicamente imposibles (precipitación negativa)
3. Impute los valores inválidos usando la climatología mensual (promedio
   histórico de ese mismo mes) calculada para cada celda.
4. NO debe intentar rellenar ni alterar las celdas fuera del polígono de
   estudio.
5. Reporte cuántas celdas del área de estudio tenían problemas y confirme
   que, tras la limpieza, no quedan NaN inesperados dentro del área de
   estudio.
```

---

## T3 — Monthly climatology and annual totals

```
Tengo un archivo NetCDF de precipitación mensual (variable "ppt", mm/mes,
dimensiones time, lat, lon) ya recortado al polígono real de 4 estados del
Noroeste de México, para 1981-2025. El archivo se llama
"ppt_noroeste_shp_1981_2025.nc" y está en la misma carpeta que el script.
Las celdas fuera del polígono son NaN/NoData.

Escribe un script en Python que calcule:
1. La climatología mensual: el promedio histórico de precipitación para
   cada uno de los 12 meses del año, promediado espacialmente sobre las
   celdas dentro del polígono (excluyendo las celdas fuera del área de
   estudio).
2. El acumulado anual de precipitación: para cada año, la SUMA de los 12
   meses (no el promedio), promediado espacialmente sobre el área de
   estudio.
3. El promedio multianual de esos acumulados anuales (1981-2025).

IMPORTANTE: al sumar los 12 meses por año, asegúrate de que las celdas
fuera del polígono de estudio den como resultado NaN (no cero) en el
acumulado — revisa el comportamiento de la función de suma que uses con
valores NaN antes de aplicarla.

Imprime los 12 valores de climatología mensual, los acumulados anuales de
los primeros y últimos 5 años, y el promedio multianual final.
```

---

## T4 — Simplified climatic water balance (ppt − pet)

```
Tengo dos archivos NetCDF ya recortados al polígono real de 4 estados del
Noroeste de México, con la misma cobertura espacial y temporal (1981-2025,
mensual): uno de precipitación ("ppt_noroeste_shp_1981_2025.nc", variable
"ppt", mm/mes) y otro de evapotranspiración potencial
("pet_noroeste_shp_1981_2025.nc", variable "pet", mm/mes). Las celdas fuera
del polígono son NaN/NoData en ambos archivos.

Escribe un script en Python que:
1. Verifique que ambos datasets tengan exactamente las mismas dimensiones
   temporales y las mismas coordenadas espaciales (lat/lon) antes de operar
   entre ellos.
2. Calcule el balance hídrico mensual como (ppt − pet).
3. Calcule el balance hídrico anual (suma de los 12 meses) para cada año,
   asegurándote de que las celdas fuera del polígono den NaN (no cero) en
   el acumulado.
4. Calcule el balance hídrico promedio multianual, promediado espacialmente
   sobre el área de estudio.
5. Imprima el balance anual de los primeros 5 años y el promedio
   multianual final, indicando si representa déficit o superávit hídrico.
```

---

## T5 — Trend and anomaly detection

```
Tengo un archivo NetCDF de precipitación mensual (variable "ppt", mm/mes,
dimensiones time, lat, lon) ya recortado al polígono real de 4 estados del
Noroeste de México, para 1981-2025. El archivo se llama
"ppt_noroeste_shp_1981_2025.nc" y está en la misma carpeta que el script.
Las celdas fuera del polígono son NaN/NoData.

Escribe un script en Python que:
1. Calcule la serie anual acumulada regional (suma de los 12 meses de cada
   año, promediada espacialmente sobre el área de estudio dentro del
   polígono). Asegúrate de que las celdas fuera del polígono no contaminen
   el resultado (revisa el comportamiento de la función de suma con NaN).
2. Calcule la tendencia lineal de esa serie (pendiente, p-valor, R²).
3. Calcule también la prueba de Mann-Kendall para detectar tendencia
   monotónica, indicando si es creciente, decreciente o sin tendencia, y
   su significancia estadística.
4. Calcule anomalías estandarizadas (z-score) de cada año respecto a la
   media y desviación estándar histórica de toda la serie.
5. Identifique y reporte los 3 años más secos y los 3 años más húmedos
   según esas anomalías.

Justifica en un comentario por qué usas (o no) cada prueba estadística
para este tipo de datos.
```

---

## T6 — Robustness to corrupted inputs

```
Necesito una función en Python para cargar de forma segura un archivo
NetCDF de datos climáticos, que se use antes de cualquier procesamiento.
Los archivos reales con los que se usará esta función se llaman, por
ejemplo, "ppt_noroeste_shp_1981_2025.nc" y "pet_noroeste_shp_1981_2025.nc"
(precipitación y evapotranspiración potencial, recortadas a un polígono
de estudio, con celdas fuera del polígono como NaN/NoData).

La función debe:
1. Verificar que el archivo exista antes de intentar abrirlo.
2. Verificar que el archivo sea un NetCDF válido (manejar el caso de que
   no lo sea, por ejemplo un archivo de texto renombrado a .nc).
3. Verificar que el archivo tenga las dimensiones esperadas: lat, lon,
   time.
4. Verificar que contenga al menos una de las variables "ppt" o "pet".
5. En cualquier caso de fallo, lanzar un error claro y descriptivo (no un
   traceback críptico de la librería subyacente).

Escribe también un pequeño bloque de prueba, en el MISMO archivo (que se
ejecute automáticamente al correr el script, sin pasos manuales
adicionales), que intente cargar:
a) un archivo que no existe
b) un archivo de texto plano renombrado a .nc

y muestre que ambos casos se manejan correctamente.
```
