# English translation of the original Spanish delivery (comments and messages only; code unchanged). Original: generated_code/phase1/t1/t1_claudeopus55.py
"""
Merges annual NetCDF files of TerraClimate monthly precipitation (1981-2025),
crops them to a geographic region and saves the result in a single NetCDF.

Requirements:
    pip install xarray netCDF4 numpy

Usage:
    python unificar_terraclimate_ppt.py
"""

from pathlib import Path
import numpy as np
import xarray as xr

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CARPETA_DATOS = Path("D:/2027-ARTICULOS/CAPITULO-SOFTWARE/ppt/")
PATRON_ARCHIVOS = "TerraClimate_ppt_*.nc"
ANIO_INICIO = 1981
ANIO_FIN = 2025

# Cropping region (Sinaloa and surrounding area)
LAT_MIN, LAT_MAX = 22.5, 32.7
LON_MIN, LON_MAX = -117.2, -105.0

VARIABLE = "ppt"
ARCHIVO_SALIDA = CARPETA_DATOS / f"TerraClimate_ppt_{ANIO_INICIO}_{ANIO_FIN}_recortado.nc"


def obtener_lista_archivos() -> list[Path]:
    """Builds and validates the list of the 45 expected annual files."""
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
            f"{len(faltantes)} expected file(s) missing in {CARPETA_DATOS}:\n"
            + "\n".join(faltantes)
        )

    print(f"Found {len(archivos)} files (expected: {ANIO_FIN - ANIO_INICIO + 1}).")
    return archivos


def cargar_y_unificar(archivos: list[Path]) -> xr.Dataset:
    """Loads the files and concatenates them into a continuous time series."""
    ds = xr.open_mfdataset(
        archivos,
        combine="by_coords",   # automatically sorts by the 'time' coordinate
        chunks={"time": 12},   # lazy loading; adjust according to available memory
        engine="netcdf4",
    )

    if VARIABLE not in ds.data_vars:
        raise KeyError(
            f"The variable '{VARIABLE}' is not in the dataset. "
            f"Available variables: {list(ds.data_vars)}"
        )

    return ds


def recortar_region(ds: xr.Dataset) -> xr.Dataset:
    """
    Crops the dataset to the region defined by LAT_MIN/MAX and LON_MIN/MAX,
    without assuming the order (ascending or descending) of the 'lat' coordinate.
    """
    lat_vals = ds["lat"].values
    lat_ascendente = lat_vals[0] < lat_vals[-1]

    if lat_ascendente:
        slice_lat = slice(LAT_MIN, LAT_MAX)
    else:
        slice_lat = slice(LAT_MAX, LAT_MIN)

    # Longitude is almost always ascending, but the same logic is applied for safety
    lon_vals = ds["lon"].values
    lon_ascendente = lon_vals[0] < lon_vals[-1]

    if lon_ascendente:
        slice_lon = slice(LON_MIN, LON_MAX)
    else:
        slice_lon = slice(LON_MAX, LON_MIN)

    ds_recortado = ds.sel(lat=slice_lat, lon=slice_lon)

    if ds_recortado.sizes["lat"] == 0 or ds_recortado.sizes["lon"] == 0:
        raise ValueError(
            "The crop came out empty. Check that the lat/lon ranges "
            "match the file's longitude convention (-180/180 vs 0/360?)."
        )

    return ds_recortado


def verificar_tiempo(ds: xr.Dataset) -> None:
    """Checks that there are no duplicate dates and that they are in chronological order."""
    tiempos = ds["time"].values

    # Duplicates
    valores_unicos, conteos = np.unique(tiempos, return_counts=True)
    duplicados = valores_unicos[conteos > 1]
    if len(duplicados) > 0:
        raise ValueError(f"Found {len(duplicados)} duplicate date(s): {duplicados}")

    # Chronological order
    if not np.all(np.diff(tiempos).astype("timedelta64[s]").astype(float) > 0):
        raise ValueError("The dates are not in strictly ascending chronological order.")

    print("Time check: no duplicates, chronological order is correct.")


def main():
    archivos = obtener_lista_archivos()
    ds = cargar_y_unificar(archivos)
    ds = recortar_region(ds)

    verificar_tiempo(ds)

    n_tiempos = ds.sizes["time"]
    n_lat = ds.sizes["lat"]
    n_lon = ds.sizes["lon"]

    print(f"\nTotal time steps: {n_tiempos}")
    print(f"Spatial dimensions: lat={n_lat}, lon={n_lon}")
    print(f"Time range: {ds['time'].values[0]} to {ds['time'].values[-1]}")

    # Save result
    print(f"\nSaving result to: {ARCHIVO_SALIDA}")
    ds.to_netcdf(ARCHIVO_SALIDA)
    print("Done.")

    ds.close()


if __name__ == "__main__":
    main()
