from pathlib import Path
import os
import xarray as xr


def cargar_netcdf_seguro(
    ruta_archivo: str | Path,
    dimensiones_requeridas: tuple[str, ...] = ("lat", "lon", "time"),
    variables_requeridas: tuple[str, ...] = ("ppt", "pet"),
) -> xr.Dataset:
    """Carga y valida un archivo NetCDF de datos climáticos.

    Lanza excepciones ValueError o FileNotFoundError con mensajes claros si el
    archivo no cumple con la estructura esperada.
    """
    path = Path(ruta_archivo)

    # 1. Verificar existencia del archivo
    if not path.is_file():
        raise FileNotFoundError(
            f"El archivo no existe o no es accesible: '{path}'"
        )

    # 2. Verificar validez como formato NetCDF
    try:
        ds = xr.open_dataset(path)
    except Exception as err:
        raise ValueError(
            f"El archivo '{path.name}' no es un NetCDF válido o está corrupto. Detalles: {err}"
        ) from err

    # 3. Verificar dimensiones esperadas
    dims_faltantes = [d for d in dimensiones_requeridas if d not in ds.dims]
    if dims_faltantes:
        ds.close()
        raise ValueError(
            f"El archivo '{path.name}' no contiene las dimensiones requeridas. "
            f"Faltantes: {dims_faltantes}. Encontradas: {list(ds.dims.keys())}"
        )

    # 4. Verificar presencia de al menos una variable esperada
    vars_presentes = [v for v in variables_requeridas if v in ds.data_vars]
    if not vars_presentes:
        ds.close()
        raise ValueError(
            f"El archivo '{path.name}' no contiene ninguna de las variables esperadas "
            f"{list(variables_requeridas)}. Variables encontradas: {list(ds.data_vars.keys())}"
        )

    return ds


def ejecutar_pruebas():
    # --- Caso a: Archivo inexistente ---
    archivo_inexistente = "archivo_fantasma.nc"

    print("--- Prueba A: Intentando cargar archivo inexistente ---")
    try:
        cargar_netcdf_seguro(archivo_inexistente)
    except (FileNotFoundError, ValueError) as e:
        print(f"ÉXITO - Error capturado correctamente:\n  -> {e}\n")

    # --- Caso b: Archivo de texto plano con extensión .nc ---
    archivo_corrupto = "datos_falsos.nc"

    # Crear archivo de texto falso
    with open(archivo_corrupto, "w", encoding="utf-8") as f:
        f.write("Este es un archivo de texto plano disfrazado de NetCDF.")

    print("--- Prueba B: Intentando cargar archivo de texto renombrado a .nc ---")
    try:
        cargar_netcdf_seguro(archivo_corrupto)
    except (FileNotFoundError, ValueError) as e:
        print(f"ÉXITO - Error capturado correctamente:\n  -> {e}\n")
    finally:
        # Limpieza del archivo temporal de prueba
        if os.path.exists(archivo_corrupto):
            os.remove(archivo_corrupto)


if __name__ == "__main__":
    ejecutar_pruebas()
