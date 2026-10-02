"""
Unifica archivos NetCDF anuales de precipitación mensual TerraClimate (1981-2025),
recorta a una región geográfica y guarda el resultado en un solo NetCDF.

Requisitos:
    pip install xarray netCDF4 numpy

Uso:
    python unificar_terraclimate_ppt.py
"""

from pathlib import Path
import numpy as np
import xarray as xr

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

CARPETA_DATOS = Path("D:/2027-ARTICULOS/CAPITULO-SOFTWARE/ppt/")
PATRON_ARCHIVOS = "TerraClimate_ppt_*.nc"
ANIO_INICIO = 1981
ANIO_FIN = 2025

# Región de recorte (Sinaloa y área circundante)
LAT_MIN, LAT_MAX = 22.5, 32.7
LON_MIN, LON_MAX = -117.2, -105.0

VARIABLE = "ppt"
ARCHIVO_SALIDA = CARPETA_DATOS / f"TerraClimate_ppt_{ANIO_INICIO}_{ANIO_FIN}_recortado.nc"


def obtener_lista_archivos() -> list[Path]:
    """Construye y valida la lista de 45 archivos anuales esperados."""
    archivos = []
    faltantes = []
    for anio in range(ANIO_INICIO, ANIO_FIN + 1):
        ruta = CARPETA_DATOS / f"TerraClimate_ppt_{anio}.nc"
        if ruta.exists():
            archivos.append(ruta)
        else:
            faltantes.append(ruta.name)

    if faltantes:
        raise FileNotFoundError(
            f"Faltan {len(faltantes)} archivo(s) esperados en {CARPETA_DATOS}:\n"
            + "\n".join(faltantes)
        )

    print(f"Se encontraron {len(archivos)} archivos (esperados: {ANIO_FIN - ANIO_INICIO + 1}).")
    return archivos


def cargar_y_unificar(archivos: list[Path]) -> xr.Dataset:
    """Carga los archivos y los concatena en una serie temporal continua."""
    ds = xr.open_mfdataset(
        archivos,
        combine="by_coords",   # ordena automáticamente por la coordenada 'time'
        chunks={"time": 12},   # carga perezosa; ajustar según memoria disponible
        engine="netcdf4",
    )

    if VARIABLE not in ds.data_vars:
        raise KeyError(
            f"La variable '{VARIABLE}' no está en el dataset. "
            f"Variables disponibles: {list(ds.data_vars)}"
        )

    return ds


def recortar_region(ds: xr.Dataset) -> xr.Dataset:
    """
    Recorta el dataset a la región definida por LAT_MIN/MAX y LON_MIN/MAX,
    sin asumir el orden (ascendente o descendente) de la coordenada 'lat'.
    """
    lat_vals = ds["lat"].values
    lat_ascendente = lat_vals[0] < lat_vals[-1]

    if lat_ascendente:
        slice_lat = slice(LAT_MIN, LAT_MAX)
    else:
        slice_lat = slice(LAT_MAX, LAT_MIN)

    # La longitud casi siempre es ascendente, pero se aplica la misma lógica por seguridad
    lon_vals = ds["lon"].values
    lon_ascendente = lon_vals[0] < lon_vals[-1]

    if lon_ascendente:
        slice_lon = slice(LON_MIN, LON_MAX)
    else:
        slice_lon = slice(LON_MAX, LON_MIN)

    ds_recortado = ds.sel(lat=slice_lat, lon=slice_lon)

    if ds_recortado.sizes["lat"] == 0 or ds_recortado.sizes["lon"] == 0:
        raise ValueError(
            "El recorte resultó vacío. Verifica que los rangos de lat/lon "
            "coincidan con la convención de longitud del archivo (¿-180/180 vs 0/360?)."
        )

    return ds_recortado


def verificar_tiempo(ds: xr.Dataset) -> None:
    """Verifica que no haya fechas duplicadas y que estén en orden cronológico."""
    tiempos = ds["time"].values

    # Duplicados
    valores_unicos, conteos = np.unique(tiempos, return_counts=True)
    duplicados = valores_unicos[conteos > 1]
    if len(duplicados) > 0:
        raise ValueError(f"Se encontraron {len(duplicados)} fecha(s) duplicada(s): {duplicados}")

    # Orden cronológico
    if not np.all(np.diff(tiempos).astype("timedelta64[s]").astype(float) > 0):
        raise ValueError("Las fechas no están en orden cronológico estrictamente ascendente.")

    print("Verificación de tiempo: sin duplicados, orden cronológico correcto.")


def main():
    archivos = obtener_lista_archivos()
    ds = cargar_y_unificar(archivos)
    ds = recortar_region(ds)

    verificar_tiempo(ds)

    n_tiempos = ds.sizes["time"]
    n_lat = ds.sizes["lat"]
    n_lon = ds.sizes["lon"]

    print(f"\nPasos de tiempo totales: {n_tiempos}")
    print(f"Dimensiones espaciales: lat={n_lat}, lon={n_lon}")
    print(f"Rango temporal: {ds['time'].values[0]} a {ds['time'].values[-1]}")

    # Guardar resultado
    print(f"\nGuardando resultado en: {ARCHIVO_SALIDA}")
    ds.to_netcdf(ARCHIVO_SALIDA)
    print("Listo.")

    ds.close()


if __name__ == "__main__":
    main()
