#!/usr/bin/env python3
# English translation of the original Spanish delivery (comments and messages only; code unchanged). Original: generated_code/phase2/t1/t1_gpt6astra.py
# -*- coding: utf-8 -*-
"""Merges monthly TerraClimate 1981-2025 and clips PPT/PET with Noroeste.shp.

Installation (in the same environment where the script will be run):
    python -m pip install numpy xarray netCDF4 geopandas rioxarray

Execution:
    python unir_recortar_terraclimate_shp.py

Paths can be changed in CONFIGURATION. Dask is not required.
A regional window of each file is read before loading its values;
then rio.clip is applied to the real polygons. The rectangle is only
a reading optimization, NOT the final mask. Only the regional data is
kept in memory, not the 90 complete global files.

The edge rule is all_touched=False: a cell is kept if its center
falls inside the polygon. No fractional cell coverage is computed.
The native resolution is kept, without interpolation. Original NaNs
(including ocean ones) remain NaN.

Documentation:
https://corteva.github.io/rioxarray/stable/examples/clip_geom.html
https://docs.xarray.dev/en/stable/generated/xarray.open_dataset.html
"""

from datetime import datetime, timezone
from pathlib import Path
import sys
import tempfile

try:
    import numpy as np
    import xarray as xr
    import geopandas as gpd
    import rioxarray  # Registers xarray's .rio accessor.
    from shapely.geometry import mapping
except ImportError as exc:
    raise SystemExit(
        f"A dependency is missing: {exc}\n"
        "Install with: python -m pip install numpy xarray netCDF4 geopandas rioxarray"
    ) from None


# --------------------------- CONFIGURATION ---------------------------
BASE = Path("D:/2027-ARTICULOS/CAPITULO-SOFTWARE")
CARPETA_PPT = BASE / "ppt"
CARPETA_PET = BASE / "pet"
SHAPEFILE = BASE / "Noroeste" / "Noroeste.shp"
ANIO_INICIAL = 1981
ANIO_FINAL = 2025

SALIDA_PPT = CARPETA_PPT / "ppt_noroeste_shp_1981_2025.nc"
SALIDA_PET = CARPETA_PET / "pet_noroeste_shp_1981_2025.nc"
CRS = "EPSG:4326"
# ---------------------------------------------------------------------


def archivos_esperados(carpeta, variable):
    """Requires one file per year; does not include other NetCDF files in the folder."""
    archivos = [
        carpeta / f"TerraClimate_{variable}_{anio}.nc"
        for anio in range(ANIO_INICIAL, ANIO_FINAL + 1)
    ]
    faltantes = [str(ruta) for ruta in archivos if not ruta.is_file()]
    if faltantes:
        raise FileNotFoundError("Missing input files:\n" + "\n".join(faltantes))
    return archivos


def leer_poligonos(ruta):
    """Reads all polygons, including separate components and interiors."""
    faltantes = [
        str(ruta.with_suffix(ext))
        for ext in (".shp", ".shx", ".dbf", ".prj")
        if not ruta.with_suffix(ext).is_file()
    ]
    if faltantes:
        raise FileNotFoundError("Incomplete shapefile:\n" + "\n".join(faltantes))
    estados = gpd.read_file(ruta)
    if estados.empty or estados.crs is None:
        raise ValueError("The shapefile is empty or has no defined CRS.")
    if estados.geometry.isna().any() or estados.geometry.is_empty.any():
        raise ValueError("The shapefile contains empty or null geometries.")
    if not estados.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        raise ValueError("The shapefile must contain only polygons.")
    if not estados.geometry.is_valid.all():
        raise ValueError("The shapefile contains invalid geometries; fix them before clipping.")
    estados = estados.to_crs(CRS)
    limites = tuple(estados.total_bounds)
    if not np.isfinite(limites).all() or not estados.geometry.is_valid.all():
        raise ValueError("The shapefile does not produce valid geometries in WGS84.")
    print(f"Shapefile: {len(estados)} polygon geometries; working CRS: {CRS}.")
    return [mapping(geom) for geom in estados.geometry], limites


def validar_tiempo(datos, inicio, fin, etiqueta):
    """Detects duplicates and missing months, sorts, and keeps the original dates."""
    if "time" not in datos.coords or datos.time.dims != ("time",):
        raise ValueError(f"{etiqueta}: a one-dimensional time coordinate is missing.")
    if bool(datos.time.isnull().any()):
        raise ValueError(f"{etiqueta}: there are null or invalid dates.")
    indice = datos.time.to_index()
    if indice.has_duplicates:
        raise ValueError(f"{etiqueta}: there are duplicate dates; they will not be removed automatically.")
    if not indice.is_monotonic_increasing:
        print(f"WARNING: {etiqueta}: dates were sorted chronologically.")
        datos = datos.sortby("time")
    try:
        meses = datos.time.dt.year.values * 12 + datos.time.dt.month.values - 1
    except (AttributeError, TypeError, ValueError):
        raise ValueError(f"{etiqueta}: time does not contain decoded monthly dates.") from None
    esperados = np.arange(inicio * 12, (fin + 1) * 12)
    if not np.array_equal(meses, esperados):
        raise ValueError(
            f"{etiqueta}: {len(esperados)} records were expected, one per month, "
            f"from January {inicio} to December {fin}; {len(meses)} were found. "
            "There are missing, repeated or out-of-period months."
        )
    return datos


def comprobar_eje(datos, nombre, etiqueta):
    """Accepts ascending or descending axes; requires a regular, finite grid."""
    if nombre not in datos.coords or datos[nombre].dims != (nombre,):
        raise ValueError(f"{etiqueta}: {nombre} must be a one-dimensional coordinate.")
    valores = datos[nombre].values
    if not np.issubdtype(valores.dtype, np.number):
        raise ValueError(f"{etiqueta}: {nombre} is not a numeric coordinate.")
    if len(valores) < 2 or not np.isfinite(valores).all():
        raise ValueError(f"{etiqueta}: the {nombre} axis is incomplete or has invalid values.")
    diferencias = np.diff(valores.astype("float64"))
    if not ((diferencias > 0).all() or (diferencias < 0).all()):
        raise ValueError(f"{etiqueta}: {nombre} has duplicates or is not monotonic.")
    paso = abs(float(valores[-1]) - float(valores[0])) / (len(valores) - 1)
    precision = np.finfo(valores.dtype).eps if valores.dtype.kind == "f" else 0
    tolerancia = max(1e-8, 4 * precision * float(np.max(np.abs(valores))))
    if not np.allclose(np.abs(diferencias), paso, rtol=1e-5, atol=tolerancia):
        raise ValueError(f"{etiqueta}: {nombre} does not form a regular grid.")
    # The tolerance above only checks regularity. The comparison
    # between files and variables, further below, IS EXACT (array_equal).
    return valores, paso


def recortar_archivo(ruta, variable, anio, geometrias, limites):
    """Loads only the regional window and applies the polygon mask."""
    try:
        with xr.open_dataset(
            ruta, engine="netcdf4", decode_cf=True,
            mask_and_scale=True, decode_coords="all", cache=False
        ) as origen:
            if variable not in origen.data_vars:
                raise ValueError(f"Does not contain the variable '{variable}'.")
            datos = origen[variable]
            if set(datos.dims) != {"time", "lat", "lon"}:
                raise ValueError(f"Dimensions of '{variable}': {datos.dims}; time, lat, lon are required.")
            datos = validar_tiempo(datos, anio, anio, ruta.name)
            lat, dy = comprobar_eje(datos, "lat", ruta.name)
            lon, dx = comprobar_eje(datos, "lon", ruta.name)
            if lat.min() < -90 or lat.max() > 90:
                raise ValueError("Latitude is not expressed in geographic degrees.")
            if lon.min() >= 0 and lon.max() <= 360:
                # Also accepts 0..360 longitudes; does not modify -180..180 ones.
                lon = np.where(lon >= 180, lon - 360, lon)
            if lon.min() < -180 or lon.max() > 180 or np.unique(lon).size != lon.size:
                raise ValueError("Longitudes are invalid or have duplicates after normalization.")
            if datos.rio.crs is not None and datos.rio.crs != CRS:
                raise ValueError(f"The NetCDF declares CRS {datos.rio.crs}; WGS84 was expected.")

            xmin, ymin, xmax, ymax = limites
            # A one-cell margin avoids losing edges due to rounding.
            iy = np.flatnonzero((lat >= ymin - dy) & (lat <= ymax + dy))
            ix = np.flatnonzero((lon >= xmin - dx) & (lon <= xmax + dx))
            if iy.size < 2 or ix.size < 2:
                raise ValueError("There is not enough overlap between the grid and the shapefile.")
            sy = slice(int(iy.min()), int(iy.max()) + 1)
            sx = slice(int(ix.min()), int(ix.max()) + 1)
            # load happens AFTER isel: the global raster is never loaded.
            regional = datos.isel(lat=sy, lon=sx).load()
            regional = regional.assign_coords(lon=("lon", lon[sx]))

        regional = regional.transpose("time", "lat", "lon")
        regional = regional.sortby("lat", ascending=False).sortby("lon")
        regional = regional.reset_coords(drop=True).astype("float32")
        # The values have already been decoded. The original packing is removed
        # to prevent a second application of scale_factor/add_offset.
        regional.encoding = {}
        for clave in (
            "_FillValue", "missing_value", "scale_factor", "add_offset", "_Unsigned",
            "valid_min", "valid_max", "valid_range", "grid_mapping", "coordinates",
        ):
            regional.attrs.pop(clave, None)
        regional = regional.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
        regional = regional.rio.write_crs(CRS).rio.write_nodata(np.nan)
        regional = regional.rio.write_coordinate_system()

        # REAL CLIP: uses all polygons and keeps their holes.
        # drop=True removes outer rows/columns; within the resulting
        # grid, cells outside the polygon become NaN.
        return regional.rio.clip(
            geometrias, crs=CRS, drop=True, all_touched=False, from_disk=False
        )
    except Exception as exc:
        raise ValueError(f"Error while processing {ruta}: {exc}") from None


def exigir_misma_grilla(a, b, etiqueta):
    for eje in ("lat", "lon"):
        if not np.array_equal(a[eje].values, b[eje].values):
            raise ValueError(
                f"{etiqueta}: the {eje} coordinates do NOT match exactly. "
                "Stopping the process; they will not be silently interpolated or aligned."
            )


def unificar_variable(archivos, variable, geometrias, limites):
    partes = []
    print(f"\nProcessing {variable.upper()}: {len(archivos)} files...")
    for numero, (anio, ruta) in enumerate(
        zip(range(ANIO_INICIAL, ANIO_FINAL + 1), archivos), start=1
    ):
        parte = recortar_archivo(ruta, variable, anio, geometrias, limites)
        if partes:
            exigir_misma_grilla(partes[0], parte, ruta.name)
            if partes[0].attrs.get("units") != parte.attrs.get("units"):
                raise ValueError(f"{ruta.name}: units differ from those of the first year.")
        partes.append(parte)
        print(f"  [{numero:02d}/{len(archivos)}] {ruta.name}: 12 months clipped.")

    # join='exact' prevents xarray from forming a spatial union with new cells.
    unido = xr.concat(partes, dim="time", coords="minimal", compat="equals", join="exact")
    partes.clear()
    unido = validar_tiempo(unido, ANIO_INICIAL, ANIO_FINAL, variable.upper())
    resultado = unido.to_dataset(name=variable)
    resultado.attrs = {
        "title": f"TerraClimate {variable}: northwest Mexico, {ANIO_INICIAL}-{ANIO_FINAL}",
        "source": "TerraClimate; original annual files",
        "region": "Baja California, Baja California Sur, Sonora and Sinaloa",
        "clip_shapefile": str(SHAPEFILE),
        "clip_method": "rioxarray.rio.clip; all_touched=False; drop=True",
        "history": f"{datetime.now(timezone.utc).isoformat()}: monthly merge and polygon clip",
    }
    print(
        f"{variable.upper()}: {resultado.sizes['time']} time steps; "
        f"{resultado.sizes['lat']} rows x {resultado.sizes['lon']} columns."
    )
    print(
        f"  {resultado.time.values[0]} to {resultado.time.values[-1]}\n"
        "  Duplicate dates: NO; chronological order: YES; complete months: YES."
    )
    return resultado


def comprobar_ppt_pet(ppt, pet):
    """Compares sizes, each coordinate and each date, with no numeric tolerance."""
    exigir_misma_grilla(ppt, pet, "PPT/PET comparison")
    if ppt.sizes["time"] != pet.sizes["time"]:
        raise ValueError("PPT and PET have a different number of time steps.")
    if not np.array_equal(ppt.time.values, pet.time.values):
        raise ValueError("PPT and PET have different dates, even though they may have the same number of months.")
    print("\nPPT / PET VERIFICATION:")
    print("  lat coordinates exactly equal: YES.")
    print("  lon coordinates exactly equal: YES.")
    print(f"  Same number of time steps: YES ({ppt.sizes['time']} in each dataset).")
    print("  Dates exactly equal: YES.")


def guardar_netcdf(dataset, variable, destino):
    """Writes a complete temporary file before replacing the previous result."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    salida = dataset.copy(deep=False)
    for nombre in salida.variables:
        salida[nombre].encoding = {}
    salida[variable].attrs = dict(salida[variable].attrs)
    salida[variable].attrs.pop("_FillValue", None)
    salida[variable].attrs["grid_mapping"] = "spatial_ref"
    codificacion = {
        variable: {
            "dtype": "float32", "_FillValue": np.float32(np.nan),
            "zlib": True, "complevel": 4, "shuffle": True,
            "chunksizes": (1, min(256, salida.sizes["lat"]), min(256, salida.sizes["lon"])),
        },
        "lat": {"_FillValue": None},
        "lon": {"_FillValue": None},
        "time": {"_FillValue": None},
    }
    with tempfile.NamedTemporaryFile(
        dir=destino.parent, prefix=destino.stem + "_", suffix=".tmp.nc", delete=False
    ) as temporal:
        ruta_temporal = Path(temporal.name)
    try:
        salida.to_netcdf(ruta_temporal, engine="netcdf4", format="NETCDF4", encoding=codificacion)
        ruta_temporal.replace(destino)
    finally:
        ruta_temporal.unlink(missing_ok=True)
    print(f"Saved: {destino}")


def main():
    archivos_ppt = archivos_esperados(CARPETA_PPT, "ppt")
    archivos_pet = archivos_esperados(CARPETA_PET, "pet")
    geometrias, limites = leer_poligonos(SHAPEFILE)
    ppt = unificar_variable(archivos_ppt, "ppt", geometrias, limites)
    pet = unificar_variable(archivos_pet, "pet", geometrias, limites)
    # No result is saved until both datasets have been verified.
    comprobar_ppt_pet(ppt, pet)
    print("\nSaving NetCDF with NaN/NoData outside the polygons...")
    guardar_netcdf(ppt, "ppt", SALIDA_PPT)
    guardar_netcdf(pet, "pet", SALIDA_PET)
    print("\nProcess completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
