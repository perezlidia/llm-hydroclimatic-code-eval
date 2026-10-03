# English translation of the original Spanish delivery (comments and messages only; code unchanged). Original: generated_code/phase2/t1/t1_claudeopus55.py
"""
Merging and polygon (shapefile) clipping of TerraClimate ppt and pet series, 1981-2025.

Workflow per variable:
  1. Opens each annual file and validates variable, dimensions and year.
  2. Normalizes coordinates: lon to [-180, 180], lat in descending order
     (north up), WITHOUT assuming the original order.
  3. Pre-selects the shapefile's bounding box (+ margin). This only
     reduces memory (the files are global); it is NOT the final clip.
  4. Concatenates the 45 years into a continuous series.
  5. Clips with the REAL POLYGON (rioxarray.clip): cells outside the
     states become NaN.
Then it checks dates (duplicates, order, monthly continuity) and the
exact equality of the grid between ppt and pet, and saves the results.

Dependencies (on Windows conda-forge is recommended):
    conda install -c conda-forge xarray netcdf4 rioxarray geopandas
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401  (registers the .rio accessor on xarray)
import xarray as xr
from shapely.geometry import mapping

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE = Path("D:/2027-ARTICULOS/CAPITULO-SOFTWARE")
CARPETAS = {"ppt": BASE / "ppt", "pet": BASE / "pet"}
SHAPEFILE = BASE / "Noroeste" / "Noroeste.shp"
SALIDA = {
    "ppt": BASE / "ppt_noroeste_shp_1981_2025.nc",
    "pet": BASE / "pet_noroeste_shp_1981_2025.nc",
}
ANIOS = range(1981, 2026)          # 1981..2025 inclusive -> 45 years
MESES_ESPERADOS = len(ANIOS) * 12  # 540
ALL_TOUCHED = False  # False: cell included if its CENTER falls inside the polygon.
                     # True: included if the polygon touches any part of the cell.


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def indice_tiempo(da):
    """Returns the time axis as a pandas.DatetimeIndex (also when it is cftime)."""
    idx = da.indexes["time"]
    if not isinstance(idx, pd.DatetimeIndex):
        idx = idx.to_datetimeindex()
    return idx


def normalizar_coordenadas(da):
    """lon to [-180, 180] sorted ascending; lat sorted descending."""
    if float(da["lon"].max()) > 180:
        da = da.assign_coords(lon=((da["lon"] + 180) % 360) - 180)
    da = da.sortby("lon", ascending=True)
    da = da.sortby("lat", ascending=False)
    return da


def cargar_shapefile(ruta):
    if not ruta.exists():
        raise FileNotFoundError(f"Shapefile not found: {ruta}")
    gdf = gpd.read_file(ruta)
    if gdf.empty:
        raise ValueError(f"The shapefile contains no geometries: {ruta}")
    if gdf.crs is None:
        print("WARNING: the shapefile declares no CRS; assuming WGS84 (EPSG:4326).")
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        print(f"WARNING: shapefile CRS = {gdf.crs}; reprojecting to EPSG:4326.")
        gdf = gdf.to_crs("EPSG:4326")
    if not gdf.is_valid.all():
        print("WARNING: there are invalid geometries; fixing them with buffer(0).")
        gdf["geometry"] = gdf.geometry.buffer(0)
    print(f"Shapefile: {len(gdf)} polygon(s), bounds {np.round(gdf.total_bounds, 3)}")
    return gdf


def cargar_serie(var, carpeta, bounds):
    """Loads the annual files, pre-selects the bounding box and concatenates."""
    minx, miny, maxx, maxy = bounds
    partes = []
    for anio in ANIOS:
        ruta = carpeta / f"TerraClimate_{var}_{anio}.nc"
        if not ruta.exists():
            raise FileNotFoundError(f"Missing file: {ruta}")

        with xr.open_dataset(ruta) as ds:
            if var not in ds.data_vars:
                raise KeyError(f"{ruta.name}: does not contain the variable '{var}'. "
                               f"Variables: {list(ds.data_vars)}")
            da = ds[var]
            faltan = {"lat", "lon", "time"} - set(da.dims)
            if faltan:
                raise ValueError(f"{ruta.name}: missing dimensions {sorted(faltan)}")

            da = normalizar_coordenadas(da)

            # 2-cell margin around the bounding box.
            res = float(np.abs(np.diff(da["lat"].values)).mean())
            m = 2 * res
            lat, lon = da["lat"], da["lon"]
            da = da.sel(
                lat=lat[(lat >= miny - m) & (lat <= maxy + m)],
                lon=lon[(lon >= minx - m) & (lon <= maxx + m)],
            )
            da = da.astype("float32").load()  # loads only the subregion

        # Validation of the year and number of months of each file.
        t = indice_tiempo(da)
        if len(t) != 12:
            print(f"WARNING: {ruta.name} has {len(t)} time steps (12 were expected).")
        if not (t.year == anio).all():
            print(f"WARNING: {ruta.name} contains dates from other years: "
                  f"{sorted(set(t.year))}")
        partes.append(da)

    # join="exact": fails if any year has a different lat/lon grid.
    serie = xr.concat(partes, dim="time", join="exact")
    print(f"[{var}] {len(partes)} files concatenated: {serie.sizes['time']} time steps.")
    return serie


def recortar_poligono(da, gdf):
    """Clip with the exact geometry; whatever lies outside becomes NaN."""
    da = da.transpose("time", "lat", "lon")
    da = da.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    da = da.rio.write_crs("EPSG:4326")
    return da.rio.clip(
        [mapping(g) for g in gdf.geometry],
        crs=gdf.crs,
        drop=True,             # reduces the grid to the polygon's extent
        all_touched=ALL_TOUCHED,
    )


def verificar_tiempo(da, var):
    """Reports duplicates, order and monthly continuity. Sorts if needed."""
    t = indice_tiempo(da)
    ok = True

    dup = t[t.duplicated()]
    if len(dup):
        ok = False
        print(f"[{var}] ERROR: {len(dup)} duplicate dates, e.g. {list(dup[:5].date)}")
    else:
        print(f"[{var}] No duplicate dates: OK")

    if t.is_monotonic_increasing:
        print(f"[{var}] Chronological order: OK")
    else:
        print(f"[{var}] WARNING: dates out of order; re-sorting them.")
        da = da.sortby("time")
        t = indice_tiempo(da)

    esperados = pd.period_range("1981-01", "2025-12", freq="M")
    presentes = t.to_period("M")
    faltan = esperados.difference(presentes)
    sobran = presentes.difference(esperados)
    if len(faltan) or len(sobran):
        ok = False
        print(f"[{var}] Continuity ERROR: {len(faltan)} months missing "
              f"{list(faltan[:5].astype(str))}, {len(sobran)} extra "
              f"{list(sobran[:5].astype(str))}")
    else:
        print(f"[{var}] Continuous monthly series 1981-01 to 2025-12: OK")

    if not ok:
        raise ValueError(f"[{var}] The time check failed; the file is not saved.")
    return da


def guardar(da, ruta):
    da = da.copy()
    da.encoding = {}               # discards the int16 packing inherited from the original
    enc = {da.name: {"dtype": "float32", "zlib": True, "complevel": 4,
                     "_FillValue": np.float32(np.nan)}}
    da.to_dataset().to_netcdf(ruta, encoding=enc)
    print(f"Saved: {ruta}")


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------
def main():
    gdf = cargar_shapefile(SHAPEFILE)
    bounds = gdf.total_bounds

    resultados = {}
    for var in ("ppt", "pet"):
        print(f"\n===== {var.upper()} =====")
        serie = cargar_serie(var, CARPETAS[var], bounds)
        serie = verificar_tiempo(serie, var)
        recortada = recortar_poligono(serie, gdf)
        resultados[var] = recortada

    ppt, pet = resultados["ppt"], resultados["pet"]

    # ---- Comparative report ------------------------------------------------
    print("\n===== CROSS-CHECK ppt vs pet =====")
    mismo_lat = np.array_equal(ppt["lat"].values, pet["lat"].values)
    mismo_lon = np.array_equal(ppt["lon"].values, pet["lon"].values)
    mismo_t = ppt.sizes["time"] == pet.sizes["time"]
    mismas_fechas = indice_tiempo(ppt).equals(indice_tiempo(pet))
    print(f"Identical latitudes:         {'YES' if mismo_lat else 'NO'}")
    print(f"Identical longitudes:        {'YES' if mismo_lon else 'NO'}")
    print(f"Same number of steps:        {'YES' if mismo_t else 'NO'} "
          f"(ppt={ppt.sizes['time']}, pet={pet.sizes['time']})")
    print(f"Exactly the same dates:      {'YES' if mismas_fechas else 'NO'}")

    for var, da in resultados.items():
        filas, cols = da.sizes["lat"], da.sizes["lon"]
        celdas_validas = int(da.isel(time=0).notnull().sum())
        print(f"\n[{var}] Time steps: {da.sizes['time']} "
              f"(expected {MESES_ESPERADOS})")
        print(f"[{var}] Clipped grid: {filas} rows x {cols} columns "
              f"= {filas * cols} cells; with data inside the polygon: {celdas_validas}")
        print(f"[{var}] Lat: {float(da.lat.max()):.4f} to {float(da.lat.min()):.4f} | "
              f"Lon: {float(da.lon.min()):.4f} to {float(da.lon.max()):.4f}")

    if not (mismo_lat and mismo_lon and mismo_t and mismas_fechas):
        raise ValueError("ppt and pet do NOT share the same grid/time; not saving them.")

    # ---- Saving -------------------------------------------------------------
    print()
    for var, da in resultados.items():
        guardar(da, SALIDA[var])


if __name__ == "__main__":
    main()
