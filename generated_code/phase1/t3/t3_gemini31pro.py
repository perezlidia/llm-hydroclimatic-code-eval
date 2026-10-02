import xarray as xr

# 1. Cargar el dataset NetCDF
ds = xr.open_dataset("ppt_noroeste_1981_2025.nc")
ppt = ds["ppt"]

# ------------------------------------------------------------------
# 1. Climatología mensual (promedio espacial de los 12 meses del año)
# ------------------------------------------------------------------
# Promediamos espacialmente sobre lat y lon, luego agrupamos por mes y promediamos
climatologia_mensual = ppt.mean(dim=["lat", "lon"]).groupby("time.month").mean(dim="time")

print("=== 1. Climatología Mensual (mm/mes) ===")
meses = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
for mes_num, val in zip(range(1, 13), climatologia_mensual.values):
    print(f"{meses[mes_num - 1]}: {val:.2f} mm")

# ------------------------------------------------------------------
# 2. Acumulado anual (Suma de los 12 meses, promediado espacialmente)
# ------------------------------------------------------------------
# Sumamos mensualmente por año ("YE" = Year End) y luego promediamos en el espacio
acumulado_anual = ppt.resample(time="YE").sum(dim="time").mean(dim=["lat", "lon"])

print("\n=== 2. Acumulados Anuales de Precipitación (mm/año) ===")
print("Primeros 5 años (1981-1985):")
for year, val in zip(acumulado_anual.time.dt.year.values[:5], acumulado_anual.values[:5]):
    print(f"  {year}: {val:.2f} mm")

print("\nÚltimos 5 años (2021-2025):")
for year, val in zip(acumulado_anual.time.dt.year.values[-5:], acumulado_anual.values[-5:]):
    print(f"  {year}: {val:.2f} mm")

# ------------------------------------------------------------------
# 3. Promedio multianual de los acumulados anuales (1981-2025)
# ------------------------------------------------------------------
promedio_multianual = acumulado_anual.mean(dim="time").item()

print(f"\n=== 3. Promedio Multianual (1981-2025) ===")
print(f"Precipitación promedio anual: {promedio_multianual:.2f} mm/año")

# Cerrar el archivo
ds.close()