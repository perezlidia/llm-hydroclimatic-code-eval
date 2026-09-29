# Phase 1 prompts — original Spanish text (rectangular clip, Python reference)

Exact prompts as sent to the models (in Spanish). An English translation is in
[`../phase1_prompts.md`](../phase1_prompts.md).

---

## T1 — Temporal merging and spatial clipping

```
Tengo 45 archivos NetCDF de precipitación mensual TerraClimate, uno por año
(1981 a 2025), nombrados TerraClimate_ppt_AAAA.nc, en la carpeta "./data/ppt/".
Cada archivo contiene una variable "ppt" con dimensiones lat, lon, time (12
meses). La cobertura es global.

Escribe un script en Python que:
1. Cargue los 45 archivos y los unifique en un solo dataset con una serie
   temporal continua de 1981 a 2025.
2. Recorte el resultado a esta región: latitud entre 22.5°N y 32.7°N,
   longitud entre -117.2° y -105.0°.
3. Verifique que no haya fechas duplicadas y que estén en orden cronológico.
4. Imprima el número total de pasos de tiempo y las dimensiones espaciales
   resultantes.
5. Guarde el resultado en un nuevo archivo NetCDF.

No asumas que las coordenadas de latitud vienen en un orden particular
(ascendente o descendente); tu código debe funcionar en ambos casos.
```

---

## T2 — Detection and handling of missing and invalid values

```
Tengo un dataset NetCDF de precipitación mensual (variable "ppt", mm/mes)
para una región que incluye tanto tierra como océano. TerraClimate es un
dataset "land-only": las celdas de océano son NaN por diseño, no representan
datos faltantes.

Necesito un script en Python que:
1. Identifique qué celdas corresponden a tierra (tienen datos válidos en
   al menos algún momento de la serie) y cuáles a océano (NaN en toda la
   serie).
2. Detecte, SOLO dentro de las celdas de tierra, valores que sean:
   a) NaN faltante inesperado
   b) Físicamente imposibles (precipitación negativa)
3. Impute los valores inválidos usando la climatología mensual (promedio
   histórico de ese mismo mes) calculada para cada celda.
4. NO debe intentar rellenar ni alterar las celdas de océano.
5. Reporte cuántas celdas de tierra tenían problemas y confirme que, tras
   la limpieza, no quedan NaN inesperados en tierra.
```

---

## T3 — Monthly climatology and annual totals

```
Tengo un dataset NetCDF de precipitación mensual (variable "ppt", mm/mes,
dimensiones lat, lon, time) para 1981-2025.

Escribe un script en Python que calcule:
1. La climatología mensual: el promedio histórico de precipitación para
   cada uno de los 12 meses del año, promediado espacialmente sobre toda
   la región (un solo número por mes).
2. El acumulado anual de precipitación: para cada año, la SUMA de los 12
   meses (no el promedio), promediado espacialmente sobre la región.
3. El promedio multianual de esos acumulados anuales (1981-2025).

Imprime los 12 valores de climatología mensual, los acumulados anuales de
los primeros y últimos 5 años, y el promedio multianual final.
```

---

## T4 — Simplified climatic water balance (ppt − pet)

```
Tengo dos datasets NetCDF con la misma cobertura espacial y temporal
(1981-2025, mensual): uno de precipitación (variable "ppt", mm/mes) y otro
de evapotranspiración potencial (variable "pet", mm/mes).

Escribe un script en Python que:
1. Verifique que ambos datasets tengan exactamente las mismas dimensiones
   temporales y las mismas coordenadas espaciales (lat/lon) antes de
   operar entre ellos. Si no coinciden, debe indicarlo claramente en vez
   de fallar de forma críptica.
2. Calcule el balance hídrico mensual como (ppt − pet).
3. Calcule el balance hídrico anual (suma de los 12 meses) para cada año.
4. Calcule el balance hídrico promedio multianual, promediado
   espacialmente sobre la región.
5. Imprima el balance anual de los primeros 5 años y el promedio
   multianual final, indicando si representa déficit o superávit hídrico.
```

---

## T5 — Trend and anomaly detection

```
Tengo una serie temporal anual de precipitación acumulada regional para
1981-2025 (45 valores, uno por año).

Escribe un script en Python que:
1. Calcule la tendencia lineal de la serie (pendiente, p-valor, R²) usando
   regresión.
2. Calcule también la prueba de Mann-Kendall para detectar tendencia
   monotónica, indicando si es creciente, decreciente o sin tendencia, y
   su significancia estadística.
3. Calcule anomalías estandarizadas (z-score) de cada año respecto a la
   media y desviación estándar histórica de toda la serie.
4. Identifique y reporte los 3 años más secos y los 3 años más húmedos
   según esas anomalías.

Justifica en un comentario por qué usas (o no) cada prueba estadística
para este tipo de datos.
```

---

## T6 — Robustness to corrupted inputs

```
Necesito una función en Python para cargar de forma segura un archivo
NetCDF de datos climáticos, que se use antes de cualquier procesamiento.

La función debe:
1. Verificar que el archivo exista antes de intentar abrirlo.
2. Verificar que el archivo sea un NetCDF válido (manejar el caso de que
   no lo sea, por ejemplo un archivo de texto renombrado a .nc).
3. Verificar que el archivo tenga las dimensiones esperadas: lat, lon,
   time.
4. Verificar que contenga al menos una de las variables "ppt" o "pet".
5. En cualquier caso de fallo, lanzar un error claro y descriptivo (no un
   traceback críptico de la librería subyacente).

Escribe también un pequeño bloque de prueba que intente cargar:
a) un archivo que no existe
b) un archivo de texto plano renombrado a .nc

y muestre que ambos casos se manejan correctamente.
```
