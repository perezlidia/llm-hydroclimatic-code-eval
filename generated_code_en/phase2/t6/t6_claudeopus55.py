# English translation of the original Spanish delivery (comments and messages only; code unchanged). Original: generated_code/phase2/t6/t6_claudeopus55.py
"""
cargar_netcdf.py

Safe loading of NetCDF climate data files (precipitation "ppt"
and/or potential evapotranspiration "pet") before any processing.

Typical usage:
    from cargar_netcdf import cargar_netcdf_seguro
    ds = cargar_netcdf_seguro("ppt_noroeste_shp_1981_2025.nc")

When this file is run directly (python cargar_netcdf.py) the tests
included at the end are run, with no additional manual steps.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import xarray as xr

DIMENSIONES_REQUERIDAS = ("lat", "lon", "time")
VARIABLES_ACEPTADAS = ("ppt", "pet")

# Binary signatures of the NetCDF formats.
_FIRMAS_NETCDF3 = (
    b"CDF\x01",  # Classic NetCDF-3
    b"CDF\x02",  # NetCDF-3 with 64-bit offsets
    b"CDF\x05",  # NetCDF-3 CDF-5 (64-bit data)
)
_FIRMA_HDF5 = b"\x89HDF\r\n\x1a\n"  # NetCDF-4 (based on HDF5)
# HDF5 allows a "userblock" at the start; the signature may be at these offsets.
_OFFSETS_HDF5 = (0, 512, 1024, 2048, 4096)


# --------------------------------------------------------------------------
# Custom exceptions: allow catching any failure with ErrorCargaNetCDF
# or, if preferred, with the equivalent standard exceptions.
# --------------------------------------------------------------------------
class ErrorCargaNetCDF(Exception):
    """Base error for any failure when loading a NetCDF file."""


class ArchivoNoEncontradoError(ErrorCargaNetCDF, FileNotFoundError):
    """The path does not exist or is not a regular file."""


class FormatoInvalidoError(ErrorCargaNetCDF, ValueError):
    """The file exists but is not a readable NetCDF."""


class EstructuraInvalidaError(ErrorCargaNetCDF, ValueError):
    """The NetCDF is valid but does not have the expected dimensions/variables."""


# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------
def _tiene_firma_netcdf(ruta: Path) -> bool:
    """Checks the initial bytes of the file to recognize NetCDF-3 or NetCDF-4."""
    with open(ruta, "rb") as f:
        if f.read(4) in _FIRMAS_NETCDF3:
            return True
        for offset in _OFFSETS_HDF5:
            f.seek(offset)
            if f.read(8) == _FIRMA_HDF5:
                return True
    return False


# --------------------------------------------------------------------------
# Main function
# --------------------------------------------------------------------------
def cargar_netcdf_seguro(
    ruta: str | Path,
    dimensiones: tuple[str, ...] = DIMENSIONES_REQUERIDAS,
    variables: tuple[str, ...] = VARIABLES_ACEPTADAS,
) -> xr.Dataset:
    """
    Opens a NetCDF climate data file after validating that it is usable.

    Validations, in order:
      1. The path exists and is a regular file (not a directory).
      2. The file is a valid NetCDF (binary signature + opening with xarray).
      3. It contains the dimensions `dimensiones` (by default lat, lon, time).
      4. It contains at least one of the variables in `variables` (by default
         ppt or pet), and those variables use the required dimensions.

    Cells outside the study polygon (NaN / _FillValue) are NOT
    considered an error: xarray converts them to NaN when opening the file.

    Returns an xarray.Dataset opened lazily. The caller
    is responsible for closing it (ds.close() or using `with`).

    Raises:
      ArchivoNoEncontradoError, FormatoInvalidoError or EstructuraInvalidaError
      (all subclasses of ErrorCargaNetCDF) with a descriptive message.
    """
    ruta = Path(ruta)

    # 1. Existence --------------------------------------------------------
    if not ruta.exists():
        raise ArchivoNoEncontradoError(
            f"The file '{ruta}' was not found. "
            f"Check the path (current working directory: '{Path.cwd()}')."
        )
    if not ruta.is_file():
        raise ArchivoNoEncontradoError(
            f"The path '{ruta}' exists but is not a file (is it a directory?)."
        )

    # 2a. Binary signature ------------------------------------------------
    try:
        if ruta.stat().st_size == 0:
            raise FormatoInvalidoError(f"The file '{ruta.name}' is empty (0 bytes).")
        es_netcdf = _tiene_firma_netcdf(ruta)
    except PermissionError:
        raise FormatoInvalidoError(
            f"There are no read permissions for the file '{ruta}'."
        ) from None

    if not es_netcdf:
        raise FormatoInvalidoError(
            f"The file '{ruta.name}' is not a valid NetCDF: its content does not "
            f"correspond to NetCDF-3 or NetCDF-4/HDF5 (it may be a file of "
            f"another type renamed with the .nc extension)."
        )

    # 2b. Actual opening with xarray (detects corrupted or truncated files)
    try:
        ds = xr.open_dataset(ruta)
    except Exception as exc:  # the underlying library can raise many types
        raise FormatoInvalidoError(
            f"The file '{ruta.name}' looks like NetCDF but could not be opened "
            f"(corrupted, truncated, or missing the netCDF4/h5netcdf engine?). "
            f"Detail: {type(exc).__name__}: {exc}"
        ) from None

    # 3 and 4. Structure --------------------------------------------------
    try:
        faltantes = [d for d in dimensiones if d not in ds.dims]
        if faltantes:
            raise EstructuraInvalidaError(
                f"The file '{ruta.name}' does not have the required dimensions "
                f"{list(dimensiones)}. Missing: {faltantes}. "
                f"Dimensions found: {list(ds.dims)}."
            )

        presentes = [v for v in variables if v in ds.data_vars]
        if not presentes:
            raise EstructuraInvalidaError(
                f"The file '{ruta.name}' does not contain any of the expected "
                f"variables {list(variables)}. "
                f"Variables found: {list(ds.data_vars)}."
            )

        for v in presentes:
            dims_faltantes = [d for d in dimensiones if d not in ds[v].dims]
            if dims_faltantes:
                raise EstructuraInvalidaError(
                    f"The variable '{v}' in '{ruta.name}' is not defined over "
                    f"the dimensions {list(dimensiones)}; it is missing {dims_faltantes}. "
                    f"Dimensions of '{v}': {list(ds[v].dims)}."
                )
    except EstructuraInvalidaError:
        ds.close()  # do not leave the file open if validation fails
        raise

    return ds


# --------------------------------------------------------------------------
# Tests: run automatically with `python cargar_netcdf.py`
# --------------------------------------------------------------------------
def _probar(descripcion: str, ruta: Path, error_esperado: type[Exception]) -> bool:
    print(f"\n[TEST] {descripcion}")
    print(f"  Path: {ruta}")
    try:
        ds = cargar_netcdf_seguro(ruta)
    except error_esperado as e:
        print(f"  OK -> {type(e).__name__} was raised as expected.")
        print(f"  Message: {e}")
        return True
    except Exception as e:
        print(f"  FAIL -> {type(e).__name__} was raised, {error_esperado.__name__} was expected.")
        print(f"  Message: {e}")
        return False
    else:
        ds.close()
        print("  FAIL -> the file was loaded without error, but it should have been rejected.")
        return False


if __name__ == "__main__":
    resultados = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # a) File that does not exist
        resultados.append(_probar(
            "a) Nonexistent file",
            tmp / "ppt_noroeste_shp_1981_2025_NO_EXISTE.nc",
            ArchivoNoEncontradoError,
        ))

        # b) Plain text file renamed to .nc
        falso = tmp / "pet_noroeste_shp_1981_2025.nc"
        falso.write_text(
            "lat,lon,time,pet\n25.8,-108.9,1981-01-01,3.2\n",
            encoding="utf-8",
        )
        resultados.append(_probar(
            "b) Plain text file renamed to .nc",
            falso,
            FormatoInvalidoError,
        ))

    total, aprobadas = len(resultados), sum(resultados)
    print(f"\nResult: {aprobadas}/{total} tests passed.")
    sys.exit(0 if aprobadas == total else 1)
