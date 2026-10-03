# English translation of the original Spanish delivery (comments and messages only; code unchanged). Original: generated_code/phase1/t6/t6_claudeopus55.py
"""
Safe loading of climate-data NetCDF files.

Validates a NetCDF file before any processing:
  1. That the file exists (and is a file, not a directory).
  2. That it is a valid NetCDF (classic format or NetCDF-4/HDF5).
  3. That it has the lat, lon and time dimensions.
  4. That it contains at least one of the variables "ppt" or "pet".
On any failure it raises a custom exception with a clear message,
instead of the traceback of the underlying library.

Requirements: xarray and a reading backend (netCDF4 recommended).
    pip install xarray netCDF4
"""

from pathlib import Path

import xarray as xr


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------
class NetCDFValidationError(Exception):
    """Base error for any validation failure of the NetCDF file."""


class ArchivoNoEncontradoError(NetCDFValidationError):
    """The file does not exist or the path does not point to a file."""


class FormatoNetCDFInvalidoError(NetCDFValidationError):
    """The file exists but is not a valid NetCDF or cannot be read."""


class DimensionesFaltantesError(NetCDFValidationError):
    """The file is missing required dimensions."""


class VariablesFaltantesError(NetCDFValidationError):
    """The file does not contain any of the required variables."""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DIMENSIONES_REQUERIDAS = ("lat", "lon", "time")
VARIABLES_ACEPTADAS = ("ppt", "pet")

# Binary signatures ("magic numbers") of the NetCDF formats
_FIRMAS_NETCDF_CLASICO = (b"CDF\x01", b"CDF\x02", b"CDF\x05")  # classic, 64-bit offset, CDF-5
_FIRMA_HDF5 = b"\x89HDF\r\n\x1a\n"                              # NetCDF-4 is based on HDF5
# HDF5 allows an initial "user block", so the signature may be at 0, 512, 1024, 2048...
_OFFSETS_HDF5 = (0, 512, 1024, 2048, 4096, 8192)


def _tiene_firma_netcdf(ruta: Path) -> bool:
    """Checks the first bytes of the file to confirm that it is NetCDF/HDF5."""
    with open(ruta, "rb") as f:
        cabecera = f.read(4)
        if cabecera in _FIRMAS_NETCDF_CLASICO:
            return True
        for offset in _OFFSETS_HDF5:
            f.seek(offset)
            if f.read(8) == _FIRMA_HDF5:
                return True
    return False


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------
def cargar_netcdf_seguro(ruta, **kwargs_open) -> xr.Dataset:
    """
    Opens and validates a climate-data NetCDF file.

    Parameters
    ----------
    ruta : str or pathlib.Path
        Path to the .nc file.
    **kwargs_open
        Additional arguments for xarray.open_dataset (e.g. chunks={"time": 12}).

    Returns
    -------
    xarray.Dataset
        Dataset opened lazily. The caller is responsible
        for closing it (ds.close() or using `with`).

    Raises
    ------
    ArchivoNoEncontradoError, FormatoNetCDFInvalidoError,
    DimensionesFaltantesError, VariablesFaltantesError
        All inherit from NetCDFValidationError.
    """
    ruta = Path(ruta)

    # 1. File existence -----------------------------------------------------
    if not ruta.exists():
        raise ArchivoNoEncontradoError(
            f"The file does not exist: '{ruta}'. Check the path and the name."
        )
    if not ruta.is_file():
        raise ArchivoNoEncontradoError(
            f"The path exists but is not a file (is it a directory?): '{ruta}'."
        )
    if ruta.stat().st_size == 0:
        raise FormatoNetCDFInvalidoError(f"The file is empty (0 bytes): '{ruta}'.")

    # 2. Validity of the NetCDF format --------------------------------------
    try:
        es_netcdf = _tiene_firma_netcdf(ruta)
    except OSError as e:
        raise FormatoNetCDFInvalidoError(
            f"The file '{ruta}' could not be read (permissions or disk): {e.strerror or e}."
        ) from None

    if not es_netcdf:
        raise FormatoNetCDFInvalidoError(
            f"The file '{ruta.name}' is not a valid NetCDF: its content does not "
            f"correspond to the classic NetCDF format or to NetCDF-4/HDF5. "
            f"It may be another type of file renamed with the .nc extension."
        )

    try:
        ds = xr.open_dataset(ruta, **kwargs_open)
    except Exception as e:  # noqa: BLE001 — translated into a descriptive error
        raise FormatoNetCDFInvalidoError(
            f"The file '{ruta.name}' has a NetCDF header but could not be opened "
            f"(it is possibly damaged, truncated, or a backend such as netCDF4 is missing). "
            f"Detail: {type(e).__name__}: {e}"
        ) from None

    # From here on, if something fails the dataset must be closed before raising.
    try:
        # 3. Required dimensions -------------------------------------------
        dims_presentes = set(ds.sizes)
        faltantes = [d for d in DIMENSIONES_REQUERIDAS if d not in dims_presentes]
        if faltantes:
            raise DimensionesFaltantesError(
                f"The file '{ruta.name}' does not have the required dimensions "
                f"{faltantes}. Dimensions found: {sorted(dims_presentes)}. "
                f"Expected: {list(DIMENSIONES_REQUERIDAS)}."
            )

        # 4. At least one accepted variable --------------------------------
        encontradas = [v for v in VARIABLES_ACEPTADAS if v in ds.data_vars]
        if not encontradas:
            raise VariablesFaltantesError(
                f"The file '{ruta.name}' does not contain any of the variables "
                f"{list(VARIABLES_ACEPTADAS)}. Variables found: "
                f"{sorted(ds.data_vars) or '(none)'}."
            )
    except NetCDFValidationError:
        ds.close()
        raise

    return ds


# ---------------------------------------------------------------------------
# Test block
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile

    def probar(descripcion, ruta):
        print(f"\n--- {descripcion} ---")
        try:
            ds = cargar_netcdf_seguro(ruta)
            print("OK: file loaded and validated.")
            print(f"    Dimensions: {dict(ds.sizes)}")
            print(f"    Variables:  {list(ds.data_vars)}")
            ds.close()
        except NetCDFValidationError as e:
            print(f"Handled error -> {type(e).__name__}:\n    {e}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # a) File that does not exist
        probar("a) Nonexistent file", tmp / "no_existe.nc")

        # b) Plain text file renamed to .nc
        falso = tmp / "texto_renombrado.nc"
        falso.write_text("This is a plain text file, not a NetCDF.\n")
        probar("b) Plain text renamed to .nc", falso)
