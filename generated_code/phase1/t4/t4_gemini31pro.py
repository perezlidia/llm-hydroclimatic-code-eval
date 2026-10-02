import sys
import numpy as np
import xarray as xr


def calcular_balance_hidrico():
    # Nombres de los archivos
    file_ppt = "ppt_noroeste_1981_2025.nc"
    file_pet = "pet_noroeste_1981_2025.nc"

    # 1. Cargar datasets
    try:
        ds_ppt = xr.open_dataset(file_ppt)
        ds_pet = xr.open_dataset(file_pet)
    except FileNotFoundError as e:
        print(f"Error: No se encontró uno de los archivos: {e}")
        sys.exit(1)

    # Variables
    ppt = ds_ppt["ppt"]
    pet = ds_pet["pet"]

    # Validar coordenadas y dimensiones espaciotemporales
    mismas_dimensiones = ppt.dims == pet.dims
    mismas_coordenadas = (
        np.array_equal(ppt["time"].values, pet["time"].values)
        and np.array_equal(ppt["lat"].values, pet["lat"].values)
        and np.array_equal(ppt["lon"].values, pet["lon"].values)
    )

    if not (mismas_dimensiones and mismas_coordenadas):
        print(
            "❌ ERROR DE VALIDACIÓN: Los datasets no coinciden exactamente en sus dimensiones o coordenadas."
        )
        if ppt.shape != pet.shape:
            print(f"   - Forma PPT: {ppt.shape} | Forma PET: {pet.shape}")
        else:
            print(
                "   - Las formas coinciden, pero los valores de las coordenadas (tiempo, latitud o longitud) varían."
            )
        sys.exit(1)

    print(
        "✅ Validación exitosa: Coordenadas espaciales y temporales idénticas.\n"
    )

    # 2. Calcular balance hídrico mensual (ppt - pet)
    balance_mensual = ppt - pet
    balance_mensual.name = "balance_hidrico"

    # 3. Calcular el balance hídrico anual (suma de los 12 meses por año)
    balance_anual = balance_mensual.resample(time="YS").sum(dim="time")

    # 4. Calcular el balance hídrico promedio multianual, promediado espacialmente
    # Promedio espacial por cada paso de tiempo, considerando las dimensiones lat/lon
    promedio_espacial_mensual = balance_mensual.mean(dim=["lat", "lon"])

    # Promedio multianual final (promedio temporal del promedio espacial)
    promedio_multianual = float(promedio_espacial_mensual.mean(dim="time"))

    # 5. Imprimir resultados de los primeros 5 años
    print("=== BALANCE HÍDRICO ANUAL PROMEDIO REGIONAL (PRIMEROS 5 AÑOS) ===")

    # Extraer los primeros 5 años y promediar espacialmente para mostrar un único valor anual regional
    balance_anual_regional = balance_anual.mean(dim=["lat", "lon"])
    anios = balance_anual_regional["time"].dt.year.values[:5]
    valores_primeros_5 = balance_anual_regional.values[:5]

    for anio, val in zip(anios, valores_primeros_5):
        estado = "Superávit" if val >= 0 else "Déficit"
        print(f"Año {anio}: {val:.2f} mm/año -> {estado}")

    print("\n=== BALANCE HÍDRICO PROMEDIO MULTIANUAL (1981-2025) ===")
    estado_multianual = "Superávit" if promedio_multianual >= 0 else "Déficit"
    print(
        f"Promedio regional mensual: {promedio_multianual:.2f} mm/mes -> {estado_multianual}"
    )

    # Cerrar datasets
    ds_ppt.close()
    ds_pet.close()


if __name__ == "__main__":
    calcular_balance_hidrico()