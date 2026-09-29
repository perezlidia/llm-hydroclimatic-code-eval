"""Carga segura de NetCDF climáticos y dos pruebas negativas automáticas.

Dependencias: python -m pip install xarray netCDF4
Ejecución:    python cargar_netcdf_climatico_seguro.py

Uso desde otro script:
    from cargar_netcdf_climatico_seguro import cargar_netcdf_seguro, ErrorNetCDF
    try:
        ds = cargar_netcdf_seguro("ppt_noroeste_shp_1981_2025.nc")
    except ErrorNetCDF as error:
        print(f"ERROR: {error}")
    else:
        print(ds)

Las rutas relativas se interpretan respecto al directorio de trabajo.
No se rellenan ni se rechazan los NaN/NoData del polígono.
Se cargan todos los datos en memoria para detectar errores de lectura antes
 de devolver el dataset; el archivo queda cerrado. Requiere RAM suficiente.
"""

from pathlib import Path
from tempfile import TemporaryDirectory


class ErrorNetCDF(ValueError):
    """Error legible de acceso, formato, estructura o lectura del NetCDF."""


def cargar_netcdf_seguro(ruta):
    """Devuelve un xarray.Dataset cargado y separado del archivo en disco.

    Exige las dimensiones lat, lon y time (permite dimensiones adicionales),
    y al menos una variable de datos llamada ppt o pet. No valida unidades,
    continuidad temporal ni plausibilidad física: son controles posteriores.
    Todos los fallos previstos se presentan como ErrorNetCDF sin encadenar
    el traceback de la biblioteca. El llamador debe capturarlo para imprimir
    únicamente el mensaje, como hacen las pruebas de este archivo.
    """
    try:
        archivo = Path(ruta).expanduser()
        if not archivo.exists():
            raise ErrorNetCDF(f"No existe el archivo: {archivo}")
        if not archivo.is_file():
            raise ErrorNetCDF(f"La ruta no corresponde a un archivo: {archivo}")
    except ErrorNetCDF:
        raise
    except (TypeError, ValueError, OSError):
        raise ErrorNetCDF("La ruta indicada es inválida o no se puede consultar.") from None

    try:
        import xarray as xr
        import netCDF4  # Motor explícito para NetCDF clásico y NetCDF4.
    except ImportError:
        raise ErrorNetCDF(
            "Faltan dependencias. Instale: python -m pip install xarray netCDF4"
        ) from None

    try:
        # No se confía en la extensión: el motor comprueba el contenido real.
        with xr.open_dataset(archivo, engine="netcdf4", decode_cf=True,
                             mask_and_scale=True) as ds:
            faltantes = {"lat", "lon", "time"} - set(ds.sizes)
            if faltantes:
                raise ErrorNetCDF(
                    f"El archivo '{archivo.name}' no tiene las dimensiones "
                    f"requeridas: {', '.join(sorted(faltantes))}. "
                    f"Dimensiones encontradas: {', '.join(ds.sizes) or '(ninguna)'}."
                )
            if not {"ppt", "pet"}.intersection(ds.data_vars):
                raise ErrorNetCDF(
                    f"El archivo '{archivo.name}' no contiene ninguna variable "
                    "de datos llamada 'ppt' o 'pet'. "
                    f"Variables encontradas: {', '.join(ds.data_vars) or '(ninguna)'}."
                )
            # Fuerza la lectura: abrir solo la cabecera no detecta todos los
            # errores de un archivo truncado o de bloques de datos dañados.
            ds.load()
        return ds
    except ErrorNetCDF:
        raise
    except PermissionError:
        raise ErrorNetCDF(f"No tiene permiso para leer: {archivo}") from None
    except MemoryError:
        raise ErrorNetCDF(
            f"No hay memoria suficiente para cargar completamente '{archivo.name}'."
        ) from None
    except Exception:
        raise ErrorNetCDF(
            f"No se pudo abrir o leer '{archivo.name}' como NetCDF climático. "
            "El contenido puede no ser NetCDF válido, estar incompleto o dañado, "
            "no ser accesible, o contener metadatos que no se pueden decodificar. "
            "Un archivo de texto renombrado a .nc no es un NetCDF."
        ) from None


def pruebas_automaticas():
    """Crea ambas entradas en una carpeta temporal; no toca archivos reales."""
    # Evita informar que la prueba de formato pasó si faltan dependencias.
    try:
        import xarray
        import netCDF4
    except ImportError:
        print("ERROR: Instale las dependencias: python -m pip install xarray netCDF4")
        return False

    correctas = 0
    with TemporaryDirectory(prefix="pruebas_netcdf_") as carpeta:
        base = Path(carpeta)
        inexistente = base / "archivo_que_no_existe.nc"
        texto = base / "texto_renombrado.nc"
        texto.write_text("Esto es texto plano, no un archivo NetCDF.\n", encoding="utf-8")
        casos = [
            ("a) Archivo inexistente", inexistente, "No existe el archivo"),
            ("b) Texto plano con extensión .nc", texto, "como NetCDF climático"),
        ]
        for titulo, ruta, mensaje_esperado in casos:
            print(f"\n{titulo}")
            try:
                cargar_netcdf_seguro(ruta)
            except ErrorNetCDF as error:
                if mensaje_esperado in str(error):
                    correctas += 1
                    print(f"[OK] Error controlado: {error}")
                else:
                    print(f"[FALLO] Error distinto del esperado: {error}")
            else:
                print("[FALLO] Se aceptó un archivo que debía rechazarse.")
    print(f"\nResultado: {correctas}/2 pruebas correctas.")
    return correctas == 2


if __name__ == "__main__":
    raise SystemExit(0 if pruebas_automaticas() else 1)
