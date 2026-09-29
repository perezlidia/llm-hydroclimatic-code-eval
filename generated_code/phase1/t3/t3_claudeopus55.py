"""
Climatología mensual y acumulado anual de precipitación (1981-2025)
Archivo: ppt_noroeste_1981_2025.nc   Variable: ppt (mm/mes)   Dims: time, lat, lon
"""
from pathlib import Path

import numpy as np
import xarray as xr

ARCHIVO = Path(__file__).resolve().parent / "ppt_noroeste_1981_2025.nc"
VARIABLE = "ppt"
ANIO_INI, ANIO_FIN = 1981, 2025

# ---------------------------------------------------------------------------
# 1. Lectura y validaciones básicas
# ---------------------------------------------------------------------------
ds = xr.open_dataset(ARCHIVO)

# Normalizar nombres de coordenadas por si vienen como latitude/longitude
renombrar = {k: v for k, v in {"latitude": "lat", "longitude": "lon"}.items()
             if k in ds.dims or k in ds.coords}
ds = ds.rename(renombrar)

ppt = ds[VARIABLE].sel(time=slice(f"{ANIO_INI}-01-01", f"{ANIO_FIN}-12-31"))

unidades = ppt.attrs.get("units", "desconocidas")
print(f"Unidades declaradas en el archivo: {unidades}")

# Los valores de relleno (p. ej. -9999) deben quedar como NaN
ppt = ppt.where(ppt >= 0)

# ---------------------------------------------------------------------------
# 2. Promedio espacial ponderado por área (cos(lat))
#    En una malla regular lat/lon, las celdas cercanas al ecuador son más
#    grandes; un promedio simple sobrerrepresentaría las celdas del norte.
#    weighted().mean() ignora NaN y renormaliza con los pesos válidos.
# ---------------------------------------------------------------------------
pesos = np.cos(np.deg2rad(ppt["lat"]))
pesos.name = "pesos"
ppt_regional = ppt.weighted(pesos).mean(dim=("lat", "lon"))  # serie mensual (mm/mes)

# ---------------------------------------------------------------------------
# 3. Climatología mensual: promedio de cada mes del calendario sobre todos
#    los años -> 12 valores (mm/mes)
# ---------------------------------------------------------------------------
climatologia = ppt_regional.groupby("time.month").mean("time")

# ---------------------------------------------------------------------------
# 4. Acumulado anual: SUMA de los 12 meses de cada año (mm/año).
#    min_count=12 hace que un año incompleto dé NaN en lugar de un total
#    artificialmente bajo.
# ---------------------------------------------------------------------------
meses_por_anio = ppt_regional.groupby("time.year").count("time")
acumulado_anual = ppt_regional.groupby("time.year").sum("time", min_count=12)

anios_incompletos = meses_por_anio.where(meses_por_anio < 12, drop=True)
if anios_incompletos.size > 0:
    print("AVISO: años con menos de 12 meses (excluidos del promedio):",
          anios_incompletos["year"].values.tolist())

# ---------------------------------------------------------------------------
# 5. Promedio multianual de los acumulados anuales (mm/año)
# ---------------------------------------------------------------------------
promedio_multianual = float(acumulado_anual.mean(skipna=True))

# ---------------------------------------------------------------------------
# 6. Resultados
# ---------------------------------------------------------------------------
nombres = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
           "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

print(f"\nClimatología mensual {ANIO_INI}-{ANIO_FIN} (promedio regional, mm/mes)")
for mes, valor in zip(climatologia["month"].values, climatologia.values):
    print(f"  {nombres[mes - 1]}: {valor:8.2f}")

anios = acumulado_anual["year"].values
valores = acumulado_anual.values

print("\nAcumulado anual - primeros 5 años (mm/año)")
for a, v in zip(anios[:5], valores[:5]):
    print(f"  {a}: {v:8.2f}")

print("\nAcumulado anual - últimos 5 años (mm/año)")
for a, v in zip(anios[-5:], valores[-5:]):
    print(f"  {a}: {v:8.2f}")

print(f"\nPromedio multianual del acumulado anual: {promedio_multianual:.2f} mm/año")

# ---------------------------------------------------------------------------
# 7. Verificación de consistencia: con años completos, la suma de las 12
#    medias mensuales debe coincidir con el promedio de los totales anuales.
# ---------------------------------------------------------------------------
suma_climatologia = float(climatologia.sum())
print(f"Suma de la climatología mensual:          {suma_climatologia:.2f} mm/año")
if not np.isclose(suma_climatologia, promedio_multianual, rtol=1e-3):
    print("AVISO: no coinciden; revisa años incompletos o NaN que varían en el tiempo.")

ds.close()
