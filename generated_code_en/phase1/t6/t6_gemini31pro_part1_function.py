# English translation of the original Spanish delivery (comments and messages only; code unchanged). Original: generated_code/phase1/t6/t6_gemini31pro_part1_function.py
from pathlib import Path
import xarray as xr


def cargar_netcdf_seguro(
    ruta_archivo: str | Path,
    dimensiones_requeridas: tuple[str, ...] = ("lat", "lon", "time"),
    variables_requeridas: tuple[str, ...] = ("ppt", "pet"),
) -> xr.Dataset:
    """Loads and validates a climate-data NetCDF file.

    Raises ValueError or FileNotFoundError exceptions with clear messages if the
    file does not comply with the expected structure.
    """
    path = Path(ruta_archivo)

    # 1. Check that the file exists
    if not path.is_file():
        raise FileNotFoundError(
            f"The file does not exist or is not accessible: '{path}'"
        )

    # 2. Check validity as NetCDF format
    try:
        ds = xr.open_dataset(path)
    except Exception as err:
        raise ValueError(
            f"The file '{path.name}' is not a valid NetCDF or is corrupt. Details: {err}"
        ) from err

    # 3. Check expected dimensions
    dims_faltantes = [d for d in dimensiones_requeridas if d not in ds.dims]
    if dims_faltantes:
        ds.close()
        raise ValueError(
            f"The file '{path.name}' does not contain the required dimensions. "
            f"Missing: {dims_faltantes}. Found: {list(ds.dims.keys())}"
        )

    # 4. Check presence of at least one expected variable
    vars_presentes = [v for v in variables_requeridas if v in ds.data_vars]
    if not vars_presentes:
        ds.close()
        raise ValueError(
            f"The file '{path.name}' does not contain any of the expected variables "
            f"{list(variables_requeridas)}. Variables found: {list(ds.data_vars.keys())}"
        )

    return ds