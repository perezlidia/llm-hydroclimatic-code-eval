"""
cargar_netcdf.py

Carga segura de archivos NetCDF de datos climáticos (precipitación "ppt"
y/o evapotranspiración potencial "pet") antes de cualquier procesamiento.

Uso típico:
    from cargar_netcdf import cargar_netcdf_seguro
    ds = cargar_netcdf_seguro("ppt_noroeste_shp_1981_2025.nc")

Al ejecutar este archivo directamente (python cargar_netcdf.py) se corren
las pruebas incluidas al final, sin pasos manuales adicionales.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import xarray as xr

DIMENSIONES_REQUERIDAS = ("lat", "lon", "time")
VARIABLES_ACEPTADAS = ("ppt", "pet")

# Firmas binarias de los formatos NetCDF.
_FIRMAS_NETCDF3 = (
    b"CDF\x01",  # NetCDF-3 clásico
    b"CDF\x02",  # NetCDF-3 con offsets de 64 bits
    b"CDF\x05",  # NetCDF-3 CDF-5 (datos de 64 bits)
)
_FIRMA_HDF5 = b"\x89HDF\r\n\x1a\n"  # NetCDF-4 (basado en HDF5)
# HDF5 permite un "userblock" al inicio; la firma puede estar en estos offsets.
_OFFSETS_HDF5 = (0, 512, 1024, 2048, 4096)


# --------------------------------------------------------------------------
# Excepciones propias: permiten capturar cualquier fallo con ErrorCargaNetCDF
# o, si se prefiere, con las excepciones estándar equivalentes.
# --------------------------------------------------------------------------
class ErrorCargaNetCDF(Exception):
    """Error base para cualquier fallo al cargar un archivo NetCDF."""


class ArchivoNoEncontradoError(ErrorCargaNetCDF, FileNotFoundError):
    """La ruta no existe o no es un archivo regular."""


class FormatoInvalidoError(ErrorCargaNetCDF, ValueError):
    """El archivo existe pero no es un NetCDF legible."""


class EstructuraInvalidaError(ErrorCargaNetCDF, ValueError):
    """El NetCDF es válido pero no tiene las dimensiones/variables esperadas."""


# --------------------------------------------------------------------------
# Funciones auxiliares
# --------------------------------------------------------------------------
def _tiene_firma_netcdf(ruta: Path) -> bool:
    """Revisa los bytes iniciales del archivo para reconocer NetCDF-3 o NetCDF-4."""
    with open(ruta, "rb") as f:
        if f.read(4) in _FIRMAS_NETCDF3:
            return True
        for offset in _OFFSETS_HDF5:
            f.seek(offset)
            if f.read(8) == _FIRMA_HDF5:
                return True
    return False


# --------------------------------------------------------------------------
# Función principal
# --------------------------------------------------------------------------
def cargar_netcdf_seguro(
    ruta: str | Path,
    dimensiones: tuple[str, ...] = DIMENSIONES_REQUERIDAS,
    variables: tuple[str, ...] = VARIABLES_ACEPTADAS,
) -> xr.Dataset:
    """
    Abre un archivo NetCDF de datos climáticos tras validar que sea utilizable.

    Validaciones, en orden:
      1. La ruta existe y es un archivo regular (no un directorio).
      2. El archivo es un NetCDF válido (firma binaria + apertura con xarray).
      3. Contiene las dimensiones `dimensiones` (por defecto lat, lon, time).
      4. Contiene al menos una de las variables en `variables` (por defecto
         ppt o pet), y esas variables usan las dimensiones requeridas.

    Las celdas fuera del polígono de estudio (NaN / _FillValue) NO se
    consideran error: xarray las convierte a NaN al abrir el archivo.

    Devuelve un xarray.Dataset abierto de forma perezosa (lazy). El llamador
    es responsable de cerrarlo (ds.close() o usar `with`).

    Lanza:
      ArchivoNoEncontradoError, FormatoInvalidoError o EstructuraInvalidaError
      (todas subclases de ErrorCargaNetCDF) con un mensaje descriptivo.
    """
    ruta = Path(ruta)

    # 1. Existencia -------------------------------------------------------
    if not ruta.exists():
        raise ArchivoNoEncontradoError(
            f"No se encontró el archivo '{ruta}'. "
            f"Verifique la ruta (directorio de trabajo actual: '{Path.cwd()}')."
        )
    if not ruta.is_file():
        raise ArchivoNoEncontradoError(
            f"La ruta '{ruta}' existe pero no es un archivo (¿es un directorio?)."
        )

    # 2a. Firma binaria ---------------------------------------------------
    try:
        if ruta.stat().st_size == 0:
            raise FormatoInvalidoError(f"El archivo '{ruta.name}' está vacío (0 bytes).")
        es_netcdf = _tiene_firma_netcdf(ruta)
    except PermissionError:
        raise FormatoInvalidoError(
            f"No hay permisos de lectura para el archivo '{ruta}'."
        ) from None

    if not es_netcdf:
        raise FormatoInvalidoError(
            f"El archivo '{ruta.name}' no es un NetCDF válido: su contenido no "
            f"corresponde a NetCDF-3 ni NetCDF-4/HDF5 (puede ser un archivo de "
            f"otro tipo renombrado con extensión .nc)."
        )

    # 2b. Apertura real con xarray (detecta archivos corruptos o truncados)
    try:
        ds = xr.open_dataset(ruta)
    except Exception as exc:  # la librería subyacente puede lanzar muchos tipos
        raise FormatoInvalidoError(
            f"El archivo '{ruta.name}' parece NetCDF pero no pudo abrirse "
            f"(¿corrupto, truncado o falta el motor netCDF4/h5netcdf?). "
            f"Detalle: {type(exc).__name__}: {exc}"
        ) from None

    # 3 y 4. Estructura ---------------------------------------------------
    try:
        faltantes = [d for d in dimensiones if d not in ds.dims]
        if faltantes:
            raise EstructuraInvalidaError(
                f"El archivo '{ruta.name}' no tiene las dimensiones requeridas "
                f"{list(dimensiones)}. Faltan: {faltantes}. "
                f"Dimensiones encontradas: {list(ds.dims)}."
            )

        presentes = [v for v in variables if v in ds.data_vars]
        if not presentes:
            raise EstructuraInvalidaError(
                f"El archivo '{ruta.name}' no contiene ninguna de las variables "
                f"esperadas {list(variables)}. "
                f"Variables encontradas: {list(ds.data_vars)}."
            )

        for v in presentes:
            dims_faltantes = [d for d in dimensiones if d not in ds[v].dims]
            if dims_faltantes:
                raise EstructuraInvalidaError(
                    f"La variable '{v}' en '{ruta.name}' no está definida sobre "
                    f"las dimensiones {list(dimensiones)}; le faltan {dims_faltantes}. "
                    f"Dimensiones de '{v}': {list(ds[v].dims)}."
                )
    except EstructuraInvalidaError:
        ds.close()  # no dejar el archivo abierto si la validación falla
        raise

    return ds


# --------------------------------------------------------------------------
# Pruebas: se ejecutan automáticamente con `python cargar_netcdf.py`
# --------------------------------------------------------------------------
def _probar(descripcion: str, ruta: Path, error_esperado: type[Exception]) -> bool:
    print(f"\n[PRUEBA] {descripcion}")
    print(f"  Ruta: {ruta}")
    try:
        ds = cargar_netcdf_seguro(ruta)
    except error_esperado as e:
        print(f"  OK -> se lanzó {type(e).__name__} como se esperaba.")
        print(f"  Mensaje: {e}")
        return True
    except Exception as e:
        print(f"  FALLO -> se lanzó {type(e).__name__}, se esperaba {error_esperado.__name__}.")
        print(f"  Mensaje: {e}")
        return False
    else:
        ds.close()
        print("  FALLO -> el archivo se cargó sin error, pero debía rechazarse.")
        return False


if __name__ == "__main__":
    resultados = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # a) Archivo que no existe
        resultados.append(_probar(
            "a) Archivo inexistente",
            tmp / "ppt_noroeste_shp_1981_2025_NO_EXISTE.nc",
            ArchivoNoEncontradoError,
        ))

        # b) Archivo de texto plano renombrado a .nc
        falso = tmp / "pet_noroeste_shp_1981_2025.nc"
        falso.write_text(
            "lat,lon,time,pet\n25.8,-108.9,1981-01-01,3.2\n",
            encoding="utf-8",
        )
        resultados.append(_probar(
            "b) Archivo de texto plano renombrado a .nc",
            falso,
            FormatoInvalidoError,
        ))

    total, aprobadas = len(resultados), sum(resultados)
    print(f"\nResultado: {aprobadas}/{total} pruebas aprobadas.")
    sys.exit(0 if aprobadas == total else 1)
