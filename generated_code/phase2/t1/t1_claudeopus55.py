"""
Unión y recorte por polígono (shapefile) de series TerraClimate ppt y pet, 1981-2025.

Flujo por variable:
  1. Abre cada archivo anual y valida variable, dimensiones y año.
  2. Normaliza coordenadas: lon a [-180, 180], lat en orden descendente
     (norte arriba), SIN suponer el orden original.
  3. Pre-selecciona la caja envolvente del shapefile (+ margen). Esto solo
     reduce memoria (los archivos son globales); NO es el recorte final.
  4. Concatena los 45 años en una serie continua.
  5. Recorta con el POLÍGONO REAL (rioxarray.clip): las celdas fuera de los
     estados quedan como NaN.
Después verifica fechas (duplicados, orden, continuidad mensual) y la
igualdad exacta de la grilla entre ppt y pet, y guarda los resultados.

Dependencias (en Windows se recomienda conda-forge):
    conda install -c conda-forge xarray netcdf4 rioxarray geopandas
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401  (registra el accesor .rio en xarray)
import xarray as xr
from shapely.geometry import mapping

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
BASE = Path("D:/2027-ARTICULOS/CAPITULO-SOFTWARE")
CARPETAS = {"ppt": BASE / "ppt", "pet": BASE / "pet"}
SHAPEFILE = BASE / "Noroeste" / "Noroeste.shp"
SALIDA = {
    "ppt": BASE / "ppt_noroeste_shp_1981_2025.nc",
    "pet": BASE / "pet_noroeste_shp_1981_2025.nc",
}
ANIOS = range(1981, 2026)          # 1981..2025 inclusive -> 45 años
MESES_ESPERADOS = len(ANIOS) * 12  # 540
ALL_TOUCHED = False  # False: celda incluida si su CENTRO cae en el polígono.
                     # True: incluida si el polígono toca cualquier parte de la celda.


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def indice_tiempo(da):
    """Devuelve el eje de tiempo como pandas.DatetimeIndex (también si es cftime)."""
    idx = da.indexes["time"]
    if not isinstance(idx, pd.DatetimeIndex):
        idx = idx.to_datetimeindex()
    return idx


def normalizar_coordenadas(da):
    """lon a [-180, 180] ordenada ascendente; lat ordenada descendente."""
    if float(da["lon"].max()) > 180:
        da = da.assign_coords(lon=((da["lon"] + 180) % 360) - 180)
    da = da.sortby("lon", ascending=True)
    da = da.sortby("lat", ascending=False)
    return da


def cargar_shapefile(ruta):
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el shapefile: {ruta}")
    gdf = gpd.read_file(ruta)
    if gdf.empty:
        raise ValueError(f"El shapefile no contiene geometrías: {ruta}")
    if gdf.crs is None:
        print("AVISO: el shapefile no declara CRS; se asume WGS84 (EPSG:4326).")
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        print(f"AVISO: CRS del shapefile = {gdf.crs}; se reproyecta a EPSG:4326.")
        gdf = gdf.to_crs("EPSG:4326")
    if not gdf.is_valid.all():
        print("AVISO: hay geometrías inválidas; se corrigen con buffer(0).")
        gdf["geometry"] = gdf.geometry.buffer(0)
    print(f"Shapefile: {len(gdf)} polígono(s), límites {np.round(gdf.total_bounds, 3)}")
    return gdf


def cargar_serie(var, carpeta, bounds):
    """Carga los archivos anuales, pre-selecciona la caja envolvente y concatena."""
    minx, miny, maxx, maxy = bounds
    partes = []
    for anio in ANIOS:
        ruta = carpeta / f"TerraClimate_{var}_{anio}.nc"
        if not ruta.exists():
            raise FileNotFoundError(f"Falta el archivo: {ruta}")

        with xr.open_dataset(ruta) as ds:
            if var not in ds.data_vars:
                raise KeyError(f"{ruta.name}: no contiene la variable '{var}'. "
                               f"Variables: {list(ds.data_vars)}")
            da = ds[var]
            faltan = {"lat", "lon", "time"} - set(da.dims)
            if faltan:
                raise ValueError(f"{ruta.name}: faltan dimensiones {sorted(faltan)}")

            da = normalizar_coordenadas(da)

            # Margen de 2 celdas alrededor de la caja envolvente.
            res = float(np.abs(np.diff(da["lat"].values)).mean())
            m = 2 * res
            lat, lon = da["lat"], da["lon"]
            da = da.sel(
                lat=lat[(lat >= miny - m) & (lat <= maxy + m)],
                lon=lon[(lon >= minx - m) & (lon <= maxx + m)],
            )
            da = da.astype("float32").load()  # carga solo la subregión

        # Validación del año y número de meses de cada archivo.
        t = indice_tiempo(da)
        if len(t) != 12:
            print(f"AVISO: {ruta.name} tiene {len(t)} pasos de tiempo (se esperaban 12).")
        if not (t.year == anio).all():
            print(f"AVISO: {ruta.name} contiene fechas de otros años: "
                  f"{sorted(set(t.year))}")
        partes.append(da)

    # join="exact": falla si algún año tiene una grilla lat/lon distinta.
    serie = xr.concat(partes, dim="time", join="exact")
    print(f"[{var}] {len(partes)} archivos concatenados: {serie.sizes['time']} pasos de tiempo.")
    return serie


def recortar_poligono(da, gdf):
    """Recorte con la geometría exacta; lo que queda fuera se vuelve NaN."""
    da = da.transpose("time", "lat", "lon")
    da = da.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    da = da.rio.write_crs("EPSG:4326")
    return da.rio.clip(
        [mapping(g) for g in gdf.geometry],
        crs=gdf.crs,
        drop=True,             # reduce la grilla a la extensión del polígono
        all_touched=ALL_TOUCHED,
    )


def verificar_tiempo(da, var):
    """Reporta duplicados, orden y continuidad mensual. Ordena si hace falta."""
    t = indice_tiempo(da)
    ok = True

    dup = t[t.duplicated()]
    if len(dup):
        ok = False
        print(f"[{var}] ERROR: {len(dup)} fechas duplicadas, p. ej. {list(dup[:5].date)}")
    else:
        print(f"[{var}] Sin fechas duplicadas: OK")

    if t.is_monotonic_increasing:
        print(f"[{var}] Orden cronológico: OK")
    else:
        print(f"[{var}] AVISO: fechas fuera de orden; se reordenan.")
        da = da.sortby("time")
        t = indice_tiempo(da)

    esperados = pd.period_range("1981-01", "2025-12", freq="M")
    presentes = t.to_period("M")
    faltan = esperados.difference(presentes)
    sobran = presentes.difference(esperados)
    if len(faltan) or len(sobran):
        ok = False
        print(f"[{var}] ERROR de continuidad: faltan {len(faltan)} meses "
              f"{list(faltan[:5].astype(str))}, sobran {len(sobran)} "
              f"{list(sobran[:5].astype(str))}")
    else:
        print(f"[{var}] Serie mensual continua 1981-01 a 2025-12: OK")

    if not ok:
        raise ValueError(f"[{var}] La verificación temporal falló; no se guarda el archivo.")
    return da


def guardar(da, ruta):
    da = da.copy()
    da.encoding = {}               # descarta el empaquetado int16 heredado del original
    enc = {da.name: {"dtype": "float32", "zlib": True, "complevel": 4,
                     "_FillValue": np.float32(np.nan)}}
    da.to_dataset().to_netcdf(ruta, encoding=enc)
    print(f"Guardado: {ruta}")


# ---------------------------------------------------------------------------
# Programa principal
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

    # ---- Reporte comparativo ------------------------------------------------
    print("\n===== VERIFICACIÓN CRUZADA ppt vs pet =====")
    mismo_lat = np.array_equal(ppt["lat"].values, pet["lat"].values)
    mismo_lon = np.array_equal(ppt["lon"].values, pet["lon"].values)
    mismo_t = ppt.sizes["time"] == pet.sizes["time"]
    mismas_fechas = indice_tiempo(ppt).equals(indice_tiempo(pet))
    print(f"Latitudes idénticas:         {'SÍ' if mismo_lat else 'NO'}")
    print(f"Longitudes idénticas:        {'SÍ' if mismo_lon else 'NO'}")
    print(f"Mismo número de pasos:       {'SÍ' if mismo_t else 'NO'} "
          f"(ppt={ppt.sizes['time']}, pet={pet.sizes['time']})")
    print(f"Mismas fechas exactas:       {'SÍ' if mismas_fechas else 'NO'}")

    for var, da in resultados.items():
        filas, cols = da.sizes["lat"], da.sizes["lon"]
        celdas_validas = int(da.isel(time=0).notnull().sum())
        print(f"\n[{var}] Pasos de tiempo: {da.sizes['time']} "
              f"(esperados {MESES_ESPERADOS})")
        print(f"[{var}] Grilla recortada: {filas} filas x {cols} columnas "
              f"= {filas * cols} celdas; con datos dentro del polígono: {celdas_validas}")
        print(f"[{var}] Lat: {float(da.lat.max()):.4f} a {float(da.lat.min()):.4f} | "
              f"Lon: {float(da.lon.min()):.4f} a {float(da.lon.max()):.4f}")

    if not (mismo_lat and mismo_lon and mismo_t and mismas_fechas):
        raise ValueError("ppt y pet NO comparten la misma grilla/tiempo; no se guardan.")

    # ---- Guardado -------------------------------------------------------------
    print()
    for var, da in resultados.items():
        guardar(da, SALIDA[var])


if __name__ == "__main__":
    main()
