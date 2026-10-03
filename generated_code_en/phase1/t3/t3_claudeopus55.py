# English translation of the original Spanish delivery (comments and messages only; code unchanged). Original: generated_code/phase1/t3/t3_claudeopus55.py
"""
Monthly climatology and annual precipitation total (1981-2025)
File: ppt_noroeste_1981_2025.nc   Variable: ppt (mm/month)   Dims: time, lat, lon
"""
from pathlib import Path

import numpy as np
import xarray as xr

ARCHIVO = Path(__file__).resolve().parent / "ppt_noroeste_1981_2025.nc"
VARIABLE = "ppt"
ANIO_INI, ANIO_FIN = 1981, 2025

# ---------------------------------------------------------------------------
# 1. Reading and basic validations
# ---------------------------------------------------------------------------
ds = xr.open_dataset(ARCHIVO)

# Normalize coordinate names in case they come as latitude/longitude
renombrar = {k: v for k, v in {"latitude": "lat", "longitude": "lon"}.items()
             if k in ds.dims or k in ds.coords}
ds = ds.rename(renombrar)

ppt = ds[VARIABLE].sel(time=slice(f"{ANIO_INI}-01-01", f"{ANIO_FIN}-12-31"))

unidades = ppt.attrs.get("units", "unknown")
print(f"Units declared in the file: {unidades}")

# Fill values (e.g. -9999) must end up as NaN
ppt = ppt.where(ppt >= 0)

# ---------------------------------------------------------------------------
# 2. Area-weighted spatial average (cos(lat))
#    On a regular lat/lon grid, cells closer to the equator are
#    larger; a simple average would overrepresent the northern cells.
#    weighted().mean() ignores NaN and renormalizes with the valid weights.
# ---------------------------------------------------------------------------
pesos = np.cos(np.deg2rad(ppt["lat"]))
pesos.name = "pesos"
ppt_regional = ppt.weighted(pesos).mean(dim=("lat", "lon"))  # monthly series (mm/month)

# ---------------------------------------------------------------------------
# 3. Monthly climatology: average of each calendar month over all
#    years -> 12 values (mm/month)
# ---------------------------------------------------------------------------
climatologia = ppt_regional.groupby("time.month").mean("time")

# ---------------------------------------------------------------------------
# 4. Annual total: SUM of the 12 months of each year (mm/year).
#    min_count=12 makes an incomplete year give NaN instead of an
#    artificially low total.
# ---------------------------------------------------------------------------
meses_por_anio = ppt_regional.groupby("time.year").count("time")
acumulado_anual = ppt_regional.groupby("time.year").sum("time", min_count=12)

anios_incompletos = meses_por_anio.where(meses_por_anio < 12, drop=True)
if anios_incompletos.size > 0:
    print("WARNING: years with fewer than 12 months (excluded from the average):",
          anios_incompletos["year"].values.tolist())

# ---------------------------------------------------------------------------
# 5. Multi-year average of the annual totals (mm/year)
# ---------------------------------------------------------------------------
promedio_multianual = float(acumulado_anual.mean(skipna=True))

# ---------------------------------------------------------------------------
# 6. Results
# ---------------------------------------------------------------------------
nombres = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

print(f"\nMonthly climatology {ANIO_INI}-{ANIO_FIN} (regional average, mm/month)")
for mes, valor in zip(climatologia["month"].values, climatologia.values):
    print(f"  {nombres[mes - 1]}: {valor:8.2f}")

anios = acumulado_anual["year"].values
valores = acumulado_anual.values

print("\nAnnual total - first 5 years (mm/year)")
for a, v in zip(anios[:5], valores[:5]):
    print(f"  {a}: {v:8.2f}")

print("\nAnnual total - last 5 years (mm/year)")
for a, v in zip(anios[-5:], valores[-5:]):
    print(f"  {a}: {v:8.2f}")

print(f"\nMulti-year average of the annual total: {promedio_multianual:.2f} mm/year")

# ---------------------------------------------------------------------------
# 7. Consistency check: with complete years, the sum of the 12
#    monthly means must match the average of the annual totals.
# ---------------------------------------------------------------------------
suma_climatologia = float(climatologia.sum())
print(f"Sum of the monthly climatology:          {suma_climatologia:.2f} mm/year")
if not np.isclose(suma_climatologia, promedio_multianual, rtol=1e-3):
    print("WARNING: they do not match; check for incomplete years or NaN that vary over time.")

ds.close()
