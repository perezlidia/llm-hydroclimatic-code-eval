import xarray as xr
import numpy as np

def clean_terraclimate_ppt(file_path, output_path, var_name='ppt'):
    # Cargar el dataset original
    ds = xr.open_dataset(file_path)
    ppt = ds[var_name]
    
    # 1. Identificar máscaras espaciales de Tierra y Océano
    # Océano: NaN en todos los tiempos. Tierra: al menos un dato válido en la serie.
    is_ocean = ppt.isnull().all(dim='time')
    is_land = ~is_ocean
    
    # 2. Detectar valores inválidos (solo dentro de la máscara de tierra)
    # a) NaN inesperados  b) Valores imposibles (precipitación negativa)
    is_nan = ppt.isnull()
    is_negative = ppt < 0
    
    # Combinar ambas condiciones solo para celdas terrestres
    invalid_mask = (is_nan | is_negative) & is_land
    num_invalid = int(invalid_mask.sum())
    
    print("=== REPORTE INICIAL ===")
    print(f"Total de celdas de océano (siempre NaN): {int(is_ocean.sum())}")
    print(f"Total de celdas de tierra (datos válidos): {int(is_land.sum())}")
    print(f"Valores inválidos detectados en tierra (NaN o <0): {num_invalid}")
    
    if num_invalid > 0:
        # Reemplazar los valores inválidos temporalmente por NaN para que no 
        # distorsionen el cálculo del promedio histórico (climatología)
        ppt_clean_base = ppt.where(~invalid_mask)
        
        # 3. Calcular climatología mensual e imputar
        # Promedio histórico mes a mes por cada celda, ignorando los NaNs
        climatology = ppt_clean_base.groupby('time.month').mean(dim='time', skipna=True)
        
        # Rellenar los huecos (NaNs creados) con la climatología correspondiente a ese mes
        ppt_imputed = ppt_clean_base.groupby('time.month').fillna(climatology)
        
        # 4. Asegurar que las celdas de océano permanezcan intactas (NaN por diseño)
        ppt_final = ppt_imputed.where(is_land)
        
        # Actualizar el dataset
        ds_cleaned = ds.copy()
        ds_cleaned[var_name] = ppt_final
    else:
        ds_cleaned = ds.copy()
        ppt_final = ppt
        print("No se detectaron valores inválidos. No se aplicó imputación.")

    # 5. Reporte de confirmación final
    remaining_nans_on_land = int((ppt_final.isnull() & is_land).sum())
    
    print("\n=== REPORTE POST-LIMPIEZA ===")
    print(f"Valores NaN restantes en tierra: {remaining_nans_on_land}")
    
    if remaining_nans_on_land > 0:
        print("⚠️ Advertencia: Algunos valores no pudieron ser imputados. Esto ocurre si "
              "una celda de tierra tiene puros NaNs en un mes específico para todos los años.")
    else:
        print("✅ Éxito: No quedan NaNs inesperados en las celdas de tierra.")
        
    # Guardar el nuevo dataset
    ds_cleaned.to_netcdf(output_path)
    print(f"\nDataset guardado en: {output_path}")
    
    return ds_cleaned

# --- EJECUCIÓN DEL SCRIPT (activada manualmente para evaluación) ---
ds_limpio = clean_terraclimate_ppt(
    file_path='ppt_noroeste_CORRUPTO_prueba_T2.nc',
    output_path='T2_gemini.nc'
)
