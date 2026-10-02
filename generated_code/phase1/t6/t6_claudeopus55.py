"""
Carga segura de archivos NetCDF de datos climáticos.

Valida un archivo NetCDF antes de cualquier procesamiento:
  1. Que el archivo exista (y sea un archivo, no un directorio).
  2. Que sea un NetCDF válido (formato clásico o NetCDF-4/HDF5).
  3. Que tenga las dimensiones lat, lon y time.
  4. Que contenga al menos una de las variables "ppt" o "pet".
Ante cualquier fallo lanza una excepción propia con un mensaje claro,
en lugar del traceback de la librería subyacente.

Requisitos: xarray y un backend de lectura (netCDF4 recomendado).
    pip install xarray netCDF4
"""

from pathlib import Path

import xarray as xr


# ---------------------------------------------------------------------------
# Excepciones propias
# ---------------------------------------------------------------------------
class NetCDFValidationError(Exception):
    """Error base para cualquier fallo de validación del archivo NetCDF."""


class ArchivoNoEncontradoError(NetCDFValidationError):
    """El archivo no existe o la ruta no apunta a un archivo."""


class FormatoNetCDFInvalidoError(NetCDFValidationError):
    """El archivo existe pero no es un NetCDF válido o no se puede leer."""


class DimensionesFaltantesError(NetCDFValidationError):
    """Al archivo le faltan dimensiones requeridas."""


class VariablesFaltantesError(NetCDFValidationError):
    """El archivo no contiene ninguna de las variables requeridas."""


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
DIMENSIONES_REQUERIDAS = ("lat", "lon", "time")
VARIABLES_ACEPTADAS = ("ppt", "pet")

# Firmas binarias ("números mágicos") de los formatos NetCDF
_FIRMAS_NETCDF_CLASICO = (b"CDF\x01", b"CDF\x02", b"CDF\x05")  # classic, 64-bit offset, CDF-5
_FIRMA_HDF5 = b"\x89HDF\r\n\x1a\n"                              # NetCDF-4 se basa en HDF5
# HDF5 permite un "user block" inicial, por lo que la firma puede estar en 0, 512, 1024, 2048...
_OFFSETS_HDF5 = (0, 512, 1024, 2048, 4096, 8192)


def _tiene_firma_netcdf(ruta: Path) -> bool:
    """Revisa los primeros bytes del archivo para confirmar que es NetCDF/HDF5."""
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
# Función principal
# ---------------------------------------------------------------------------
def cargar_netcdf_seguro(ruta, **kwargs_open) -> xr.Dataset:
    """
    Abre y valida un archivo NetCDF de datos climáticos.

    Parámetros
    ----------
    ruta : str o pathlib.Path
        Ruta al archivo .nc.
    **kwargs_open
        Argumentos adicionales para xarray.open_dataset (p. ej. chunks={"time": 12}).

    Devuelve
    --------
    xarray.Dataset
        Dataset abierto de forma perezosa (lazy). El llamador es responsable
        de cerrarlo (ds.close() o usar `with`).

    Lanza
    -----
    ArchivoNoEncontradoError, FormatoNetCDFInvalidoError,
    DimensionesFaltantesError, VariablesFaltantesError
        Todas heredan de NetCDFValidationError.
    """
    ruta = Path(ruta)

    # 1. Existencia del archivo -------------------------------------------
    if not ruta.exists():
        raise ArchivoNoEncontradoError(
            f"El archivo no existe: '{ruta}'. Verifique la ruta y el nombre."
        )
    if not ruta.is_file():
        raise ArchivoNoEncontradoError(
            f"La ruta existe pero no es un archivo (¿es un directorio?): '{ruta}'."
        )
    if ruta.stat().st_size == 0:
        raise FormatoNetCDFInvalidoError(f"El archivo está vacío (0 bytes): '{ruta}'.")

    # 2. Validez del formato NetCDF ---------------------------------------
    try:
        es_netcdf = _tiene_firma_netcdf(ruta)
    except OSError as e:
        raise FormatoNetCDFInvalidoError(
            f"No se pudo leer el archivo '{ruta}' (permisos o disco): {e.strerror or e}."
        ) from None

    if not es_netcdf:
        raise FormatoNetCDFInvalidoError(
            f"El archivo '{ruta.name}' no es un NetCDF válido: su contenido no "
            f"corresponde al formato NetCDF clásico ni a NetCDF-4/HDF5. "
            f"Es posible que sea otro tipo de archivo renombrado con extensión .nc."
        )

    try:
        ds = xr.open_dataset(ruta, **kwargs_open)
    except Exception as e:  # noqa: BLE001 — se traduce a un error descriptivo
        raise FormatoNetCDFInvalidoError(
            f"El archivo '{ruta.name}' tiene cabecera NetCDF pero no pudo abrirse "
            f"(posiblemente está dañado, truncado o falta un backend como netCDF4). "
            f"Detalle: {type(e).__name__}: {e}"
        ) from None

    # A partir de aquí, si algo falla hay que cerrar el dataset antes de lanzar.
    try:
        # 3. Dimensiones requeridas ---------------------------------------
        dims_presentes = set(ds.sizes)
        faltantes = [d for d in DIMENSIONES_REQUERIDAS if d not in dims_presentes]
        if faltantes:
            raise DimensionesFaltantesError(
                f"El archivo '{ruta.name}' no tiene las dimensiones requeridas "
                f"{faltantes}. Dimensiones encontradas: {sorted(dims_presentes)}. "
                f"Se esperaban: {list(DIMENSIONES_REQUERIDAS)}."
            )

        # 4. Al menos una variable aceptada -------------------------------
        encontradas = [v for v in VARIABLES_ACEPTADAS if v in ds.data_vars]
        if not encontradas:
            raise VariablesFaltantesError(
                f"El archivo '{ruta.name}' no contiene ninguna de las variables "
                f"{list(VARIABLES_ACEPTADAS)}. Variables encontradas: "
                f"{sorted(ds.data_vars) or '(ninguna)'}."
            )
    except NetCDFValidationError:
        ds.close()
        raise

    return ds


# ---------------------------------------------------------------------------
# Bloque de prueba
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile

    def probar(descripcion, ruta):
        print(f"\n--- {descripcion} ---")
        try:
            ds = cargar_netcdf_seguro(ruta)
            print("OK: archivo cargado y validado.")
            print(f"    Dimensiones: {dict(ds.sizes)}")
            print(f"    Variables:   {list(ds.data_vars)}")
            ds.close()
        except NetCDFValidationError as e:
            print(f"Error controlado -> {type(e).__name__}:\n    {e}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # a) Archivo que no existe
        probar("a) Archivo inexistente", tmp / "no_existe.nc")

        # b) Archivo de texto plano renombrado a .nc
        falso = tmp / "texto_renombrado.nc"
        falso.write_text("Esto es un archivo de texto plano, no un NetCDF.\n")
        probar("b) Texto plano renombrado a .nc", falso)
