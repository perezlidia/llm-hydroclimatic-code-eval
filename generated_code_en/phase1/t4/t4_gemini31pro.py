# English translation of the original Spanish delivery (comments and messages only; code unchanged). Original: generated_code/phase1/t4/t4_gemini31pro.py
import sys
import numpy as np
import xarray as xr


def calcular_balance_hidrico():
    # File names
    file_ppt = "ppt_noroeste_1981_2025.nc"
    file_pet = "pet_noroeste_1981_2025.nc"

    # 1. Load datasets
    try:
        ds_ppt = xr.open_dataset(file_ppt)
        ds_pet = xr.open_dataset(file_pet)
    except FileNotFoundError as e:
        print(f"Error: One of the files was not found: {e}")
        sys.exit(1)

    # Variables
    ppt = ds_ppt["ppt"]
    pet = ds_pet["pet"]

    # Validate spatiotemporal coordinates and dimensions
    mismas_dimensiones = ppt.dims == pet.dims
    mismas_coordenadas = (
        np.array_equal(ppt["time"].values, pet["time"].values)
        and np.array_equal(ppt["lat"].values, pet["lat"].values)
        and np.array_equal(ppt["lon"].values, pet["lon"].values)
    )

    if not (mismas_dimensiones and mismas_coordenadas):
        print(
            "❌ VALIDATION ERROR: The datasets do not match exactly in their dimensions or coordinates."
        )
        if ppt.shape != pet.shape:
            print(f"   - PPT shape: {ppt.shape} | PET shape: {pet.shape}")
        else:
            print(
                "   - The shapes match, but the coordinate values (time, latitude or longitude) differ."
            )
        sys.exit(1)

    print(
        "✅ Validation successful: Identical spatial and temporal coordinates.\n"
    )

    # 2. Compute monthly water balance (ppt - pet)
    balance_mensual = ppt - pet
    balance_mensual.name = "balance_hidrico"

    # 3. Compute the annual water balance (sum of the 12 months per year)
    balance_anual = balance_mensual.resample(time="YS").sum(dim="time")

    # 4. Compute the multi-year mean water balance, spatially averaged
    # Spatial mean for each time step, considering the lat/lon dimensions
    promedio_espacial_mensual = balance_mensual.mean(dim=["lat", "lon"])

    # Final multi-year mean (temporal mean of the spatial mean)
    promedio_multianual = float(promedio_espacial_mensual.mean(dim="time"))

    # 5. Print results for the first 5 years
    print("=== REGIONAL MEAN ANNUAL WATER BALANCE (FIRST 5 YEARS) ===")

    # Extract the first 5 years and average spatially to show a single regional annual value
    balance_anual_regional = balance_anual.mean(dim=["lat", "lon"])
    anios = balance_anual_regional["time"].dt.year.values[:5]
    valores_primeros_5 = balance_anual_regional.values[:5]

    for anio, val in zip(anios, valores_primeros_5):
        estado = "Surplus" if val >= 0 else "Deficit"
        print(f"Year {anio}: {val:.2f} mm/year -> {estado}")

    print("\n=== MULTI-YEAR MEAN WATER BALANCE (1981-2025) ===")
    estado_multianual = "Surplus" if promedio_multianual >= 0 else "Deficit"
    print(
        f"Regional monthly mean: {promedio_multianual:.2f} mm/month -> {estado_multianual}"
    )

    # Close datasets
    ds_ppt.close()
    ds_pet.close()


if __name__ == "__main__":
    calcular_balance_hidrico()