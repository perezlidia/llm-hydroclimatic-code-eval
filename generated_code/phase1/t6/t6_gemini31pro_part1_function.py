from pathlib import Path
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