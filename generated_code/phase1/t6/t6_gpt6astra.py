"""Tarea 6. Carga de NetCDF climático con errores descriptivos.

Dependencias: python -m pip install xarray netCDF4
Pruebas:      python cargar_netcdf_seguro_T6.py

La función devuelve un xarray.Dataset cargado en memoria y cierra el archivo.
Por tanto, el contenido descomprimido debe caber en la memoria disponible.
Se requieren lat, lon y time; se permiten dimensiones adicionales y cualquier
orden. Basta con una variable de datos ppt o pet. No se evalúa aquí la calidad
física de sus valores ni se imputan NaN.

Referencia: https://docs.xarray.dev/en/stable/generated/xarray.Dataset.load.html
"""

from pathlib import Path
from tempfile import TemporaryDirectory


class ErrorNetCDF(Exception):
    """Error de acceso, formato, estructura o lectura del archivo climático."""


def cargar_netcdf_seguro(ruta):
    """Valida y carga un NetCDF local; ante un fallo lanza ErrorNetCDF.

    El llamador debe capturar ErrorNetCDF e imprimir su mensaje si desea evitar
    un traceback. Se suprimen las cadenas de excepciones de las bibliotecas.
    El archivo se abre en modo lectura y no se modifica.
    """
    try:
        archivo = Path(ruta).expanduser()
    except (TypeError, ValueError, RuntimeError):
        raise ErrorNetCDF("Ruta inválida: indique un nombre de archivo local.") from None

    # 1. Comprobar la ruta antes de intentar abrir el archivo.
    try:
        if not archivo.exists():
            raise ErrorNetCDF(f"No existe el archivo: {archivo}")
        if not archivo.is_file():
            raise ErrorNetCDF(f"La ruta no corresponde a un archivo: {archivo}")
    except OSError:
        raise ErrorNetCDF(f"No se puede consultar la ruta: {archivo}") from None

    try:
        import xarray as xr
        import netCDF4  # Verifica que el motor de lectura esté instalado.
    except ImportError:
        raise ErrorNetCDF(
            "Faltan dependencias o no pueden importarse. Ejecute: "
            "python -m pip install xarray netCDF4"
        ) from None

    etapa = "abrir"
    try:
        # 2. El motor analiza el contenido; la extensión .nc no basta.
        with xr.open_dataset(archivo, engine="netcdf4") as ds:
            # 3. Exigir las dimensiones, independientemente de su orden.
            faltantes = sorted({"lat", "lon", "time"} - set(ds.sizes))
            if faltantes:
                raise ErrorNetCDF(
                    f"Dimensiones faltantes en '{archivo}': "
                    f"{', '.join(faltantes)}. "
                    f"Dimensiones disponibles: {list(ds.sizes)}."
                )

            # 4. Basta una de las dos variables de datos.
            if not {"ppt", "pet"}.intersection(ds.data_vars):
                raise ErrorNetCDF(
                    f"El archivo '{archivo}' debe contener al menos una "
                    f"variable de datos 'ppt' o 'pet'. "
                    f"Variables disponibles: {list(ds.data_vars)}."
                )

            # Leer ahora permite capturar errores que una apertura diferida
            # podría dejar para una operación posterior.
            etapa = "leer los datos de"
            ds.load()

        return ds  # Los datos siguen disponibles aunque el archivo ya cerró.

    except ErrorNetCDF:
        raise
    except FileNotFoundError:
        raise ErrorNetCDF(f"El archivo dejó de estar disponible: {archivo}") from None
    except PermissionError:
        raise ErrorNetCDF(f"No hay permiso para leer el archivo: {archivo}") from None
    except MemoryError:
        raise ErrorNetCDF(
            f"Memoria insuficiente para cargar completamente '{archivo}'."
        ) from None
    except Exception:
        # Se limita esta captura a la apertura, validación y lectura del NetCDF.
        detalle = (
            "No es un NetCDF válido, está dañado, no es accesible "
            "o contiene metadatos que no pueden decodificarse."
            if etapa == "abrir"
            else "Alguna variable no pudo leerse o decodificarse; "
                 "el archivo puede estar dañado."
        )
        raise ErrorNetCDF(
            f"No se pudo {etapa} '{archivo}'. {detalle}"
        ) from None


if __name__ == "__main__":
    # Carpeta temporal: no sobrescribe archivos del usuario y se elimina al salir.
    with TemporaryDirectory(prefix="prueba_netcdf_") as carpeta:
        base = Path(carpeta)
        inexistente = base / "no_existe.nc"

        texto = base / "texto_plano.txt"
        texto.write_text("Esto es texto plano, no un NetCDF.\n", encoding="utf-8")
        falso_netcdf = texto.rename(base / "texto_renombrado.nc")

        casos = [
            ("a) Archivo inexistente", inexistente, "No existe el archivo"),
            ("b) Texto renombrado a .nc", falso_netcdf, "No se pudo abrir"),
        ]
        for nombre, ruta, mensaje_esperado in casos:
            try:
                cargar_netcdf_seguro(ruta)
            except ErrorNetCDF as error:
                if mensaje_esperado not in str(error):
                    raise SystemExit(f"{nombre}: fallo distinto al esperado: {error}")
                print(f"{nombre}: ERROR CONTROLADO\n  {error}\n")
            else:
                raise SystemExit(f"PRUEBA FALLIDA: se aceptó '{ruta.name}'.")

        print("Ambas pruebas se manejaron correctamente, sin traceback.")
