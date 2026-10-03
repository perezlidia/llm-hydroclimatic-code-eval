# English translation of the original Spanish delivery (comments and messages only; code unchanged). Original: generated_code/phase2/t6/t6_gpt6astra.py
"""Safe loading of climate NetCDF files and two automatic negative tests.

Dependencies: python -m pip install xarray netCDF4
Execution:    python cargar_netcdf_climatico_seguro.py

Usage from another script:
    from cargar_netcdf_climatico_seguro import cargar_netcdf_seguro, ErrorNetCDF
    try:
        ds = cargar_netcdf_seguro("ppt_noroeste_shp_1981_2025.nc")
    except ErrorNetCDF as error:
        print(f"ERROR: {error}")
    else:
        print(ds)

Relative paths are interpreted with respect to the working directory.
The NaN/NoData values of the polygon are neither filled nor rejected.
All data are loaded into memory to detect read errors before
 returning the dataset; the file is left closed. Requires sufficient RAM.
"""

from pathlib import Path
from tempfile import TemporaryDirectory


class ErrorNetCDF(ValueError):
    """Readable error for NetCDF access, format, structure or reading."""


def cargar_netcdf_seguro(ruta):
    """Returns an xarray.Dataset loaded and detached from the file on disk.

    Requires the dimensions lat, lon and time (allows additional dimensions),
    and at least one data variable named ppt or pet. Does not validate units,
    temporal continuity or physical plausibility: those are later checks.
    All anticipated failures are presented as ErrorNetCDF without chaining
    the library traceback. The caller must catch it to print
    only the message, as the tests in this file do.
    """
    try:
        archivo = Path(ruta).expanduser()
        if not archivo.exists():
            raise ErrorNetCDF(f"No existe el archivo: {archivo}")  # kept in Spanish: matched by the test below ("The file does not exist: ...")
        if not archivo.is_file():
            raise ErrorNetCDF(f"The path does not correspond to a file: {archivo}")
    except ErrorNetCDF:
        raise
    except (TypeError, ValueError, OSError):
        raise ErrorNetCDF("The given path is invalid or cannot be queried.") from None

    try:
        import xarray as xr
        import netCDF4  # Explicit engine for classic NetCDF and NetCDF4.
    except ImportError:
        raise ErrorNetCDF(
            "Missing dependencies. Install: python -m pip install xarray netCDF4"
        ) from None

    try:
        # The extension is not trusted: the engine checks the actual content.
        with xr.open_dataset(archivo, engine="netcdf4", decode_cf=True,
                             mask_and_scale=True) as ds:
            faltantes = {"lat", "lon", "time"} - set(ds.sizes)
            if faltantes:
                raise ErrorNetCDF(
                    f"The file '{archivo.name}' does not have the required "
                    f"dimensions: {', '.join(sorted(faltantes))}. "
                    f"Dimensions found: {', '.join(ds.sizes) or '(none)'}."
                )
            if not {"ppt", "pet"}.intersection(ds.data_vars):
                raise ErrorNetCDF(
                    f"The file '{archivo.name}' does not contain any data "
                    "variable named 'ppt' or 'pet'. "
                    f"Variables found: {', '.join(ds.data_vars) or '(none)'}."
                )
            # Forces reading: opening only the header does not detect all
            # errors of a truncated file or of damaged data blocks.
            ds.load()
        return ds
    except ErrorNetCDF:
        raise
    except PermissionError:
        raise ErrorNetCDF(f"You do not have permission to read: {archivo}") from None
    except MemoryError:
        raise ErrorNetCDF(
            f"There is not enough memory to fully load '{archivo.name}'."
        ) from None
    except Exception:
        raise ErrorNetCDF(
            f"No se pudo abrir o leer '{archivo.name}' como NetCDF climático. "  # kept in Spanish: matched by the test below ("Could not open or read ... as climate NetCDF.")
            "The content may not be valid NetCDF, may be incomplete or damaged, "
            "may not be accessible, or may contain metadata that cannot be decoded. "
            "A text file renamed to .nc is not a NetCDF."
        ) from None


def pruebas_automaticas():
    """Creates both inputs in a temporary folder; does not touch real files."""
    # Avoids reporting that the format test passed if dependencies are missing.
    try:
        import xarray
        import netCDF4
    except ImportError:
        print("ERROR: Install the dependencies: python -m pip install xarray netCDF4")
        return False

    correctas = 0
    with TemporaryDirectory(prefix="pruebas_netcdf_") as carpeta:
        base = Path(carpeta)
        inexistente = base / "archivo_que_no_existe.nc"
        texto = base / "texto_renombrado.nc"
        texto.write_text("This is plain text, not a NetCDF file.\n", encoding="utf-8")
        casos = [
            ("a) Nonexistent file", inexistente, "No existe el archivo"),  # expected substring kept in Spanish
            ("b) Plain text with .nc extension", texto, "como NetCDF climático"),  # expected substring kept in Spanish
        ]
        for titulo, ruta, mensaje_esperado in casos:
            print(f"\n{titulo}")
            try:
                cargar_netcdf_seguro(ruta)
            except ErrorNetCDF as error:
                if mensaje_esperado in str(error):
                    correctas += 1
                    print(f"[OK] Handled error: {error}")
                else:
                    print(f"[FAIL] Error different from the expected one: {error}")
            else:
                print("[FAIL] A file that should have been rejected was accepted.")
    print(f"\nResult: {correctas}/2 tests correct.")
    return correctas == 2


if __name__ == "__main__":
    raise SystemExit(0 if pruebas_automaticas() else 1)
