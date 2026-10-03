# English translation of the original Spanish delivery (comments and messages only; code unchanged). Original: generated_code/phase1/t6/t6_gpt6astra.py
"""Task 6. Loading of climate NetCDF with descriptive errors.

Dependencies: python -m pip install xarray netCDF4
Tests:        python cargar_netcdf_seguro_T6.py

The function returns an xarray.Dataset loaded into memory and closes the file.
Therefore, the decompressed content must fit in the available memory.
lat, lon and time are required; additional dimensions and any
order are allowed. One data variable ppt or pet is enough. The physical quality
of its values is not evaluated here, nor are NaN imputed.

Reference: https://docs.xarray.dev/en/stable/generated/xarray.Dataset.load.html
"""

from pathlib import Path
from tempfile import TemporaryDirectory


class ErrorNetCDF(Exception):
    """Access, format, structure or reading error of the climate file."""


def cargar_netcdf_seguro(ruta):
    """Validates and loads a local NetCDF; on failure raises ErrorNetCDF.

    The caller must catch ErrorNetCDF and print its message if it wants to avoid
    a traceback. Library exception chains are suppressed.
    The file is opened in read mode and is not modified.
    """
    try:
        archivo = Path(ruta).expanduser()
    except (TypeError, ValueError, RuntimeError):
        raise ErrorNetCDF("Invalid path: provide a local file name.") from None

    # 1. Check the path before trying to open the file.
    try:
        if not archivo.exists():
            raise ErrorNetCDF(f"No existe el archivo: {archivo}")  # kept in Spanish (matched by the test below): "The file does not exist"
        if not archivo.is_file():
            raise ErrorNetCDF(f"The path does not correspond to a file: {archivo}")
    except OSError:
        raise ErrorNetCDF(f"The path cannot be queried: {archivo}") from None

    try:
        import xarray as xr
        import netCDF4  # Verifies that the reading engine is installed.
    except ImportError:
        raise ErrorNetCDF(
            "Dependencies are missing or cannot be imported. Run: "
            "python -m pip install xarray netCDF4"
        ) from None

    etapa = "abrir"  # kept in Spanish (compared below and matched by the test): "open"
    try:
        # 2. The engine analyzes the content; the .nc extension is not enough.
        with xr.open_dataset(archivo, engine="netcdf4") as ds:
            # 3. Require the dimensions, regardless of their order.
            faltantes = sorted({"lat", "lon", "time"} - set(ds.sizes))
            if faltantes:
                raise ErrorNetCDF(
                    f"Missing dimensions in '{archivo}': "
                    f"{', '.join(faltantes)}. "
                    f"Available dimensions: {list(ds.sizes)}."
                )

            # 4. One of the two data variables is enough.
            if not {"ppt", "pet"}.intersection(ds.data_vars):
                raise ErrorNetCDF(
                    f"The file '{archivo}' must contain at least one "
                    f"data variable 'ppt' or 'pet'. "
                    f"Available variables: {list(ds.data_vars)}."
                )

            # Reading now makes it possible to catch errors that a lazy opening
            # could leave for a later operation.
            etapa = "leer los datos de"  # kept in Spanish (inserted into the message below): "read the data of"
            ds.load()

        return ds  # The data remain available even though the file has already closed.

    except ErrorNetCDF:
        raise
    except FileNotFoundError:
        raise ErrorNetCDF(f"The file is no longer available: {archivo}") from None
    except PermissionError:
        raise ErrorNetCDF(f"No permission to read the file: {archivo}") from None
    except MemoryError:
        raise ErrorNetCDF(
            f"Insufficient memory to fully load '{archivo}'."
        ) from None
    except Exception:
        # This catch is limited to opening, validating and reading the NetCDF.
        detalle = (
            "It is not a valid NetCDF, it is damaged, it is not accessible "
            "or it contains metadata that cannot be decoded."
            if etapa == "abrir"
            else "Some variable could not be read or decoded; "
                 "the file may be damaged."
        )
        raise ErrorNetCDF(
            f"No se pudo {etapa} '{archivo}'. {detalle}"  # "No se pudo" kept in Spanish (matched by the test below): "Could not <stage>"
        ) from None


if __name__ == "__main__":
    # Temporary folder: does not overwrite user files and is deleted on exit.
    with TemporaryDirectory(prefix="prueba_netcdf_") as carpeta:
        base = Path(carpeta)
        inexistente = base / "no_existe.nc"

        texto = base / "texto_plano.txt"
        texto.write_text("This is plain text, not a NetCDF.\n", encoding="utf-8")
        falso_netcdf = texto.rename(base / "texto_renombrado.nc")

        casos = [
            ("a) Nonexistent file", inexistente, "No existe el archivo"),  # expected text kept in Spanish (compared with the error message)
            ("b) Text renamed to .nc", falso_netcdf, "No se pudo abrir"),  # expected text kept in Spanish (compared with the error message)
        ]
        for nombre, ruta, mensaje_esperado in casos:
            try:
                cargar_netcdf_seguro(ruta)
            except ErrorNetCDF as error:
                if mensaje_esperado not in str(error):
                    raise SystemExit(f"{nombre}: failure different from the expected one: {error}")
                print(f"{nombre}: HANDLED ERROR\n  {error}\n")
            else:
                raise SystemExit(f"TEST FAILED: '{ruta.name}' was accepted.")

        print("Both tests were handled correctly, without traceback.")
