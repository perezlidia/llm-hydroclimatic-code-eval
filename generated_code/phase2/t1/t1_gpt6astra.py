#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Une TerraClimate mensual de 1981-2025 y recorta PPT/PET con Noroeste.shp.

Instalación (en el mismo entorno donde se ejecutará el script):
    python -m pip install numpy xarray netCDF4 geopandas rioxarray

Ejecución:
    python unir_recortar_terraclimate_shp.py

Las rutas se pueden modificar en CONFIGURACIÓN. No necesita Dask.
Se lee una ventana regional de cada archivo antes de cargar sus valores;
después se aplica rio.clip a los polígonos reales. El rectángulo es únicamente
una optimización de lectura, NO la máscara final. Solo se mantienen en memoria
los datos regionales, no los 90 archivos globales completos.

La regla de borde es all_touched=False: se conserva una celda si su centro
queda dentro del polígono. No se calcula cobertura fraccional de las celdas.
Se conserva la resolución nativa, sin interpolación. Los NaN originales
(incluidos los del océano) permanecen como NaN.

Documentación:
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
    import rioxarray  # Registra el accesor .rio de xarray.
    from shapely.geometry import mapping
except ImportError as exc:
    raise SystemExit(
        f"Falta una dependencia: {exc}\n"
        "Instala con: python -m pip install numpy xarray netCDF4 geopandas rioxarray"
    ) from None


# --------------------------- CONFIGURACIÓN ---------------------------
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
    """Exige un archivo por cada año; no incluye otros NetCDF de la carpeta."""
    archivos = [
        carpeta / f"TerraClimate_{variable}_{anio}.nc"
        for anio in range(ANIO_INICIAL, ANIO_FINAL + 1)
    ]
    faltantes = [str(ruta) for ruta in archivos if not ruta.is_file()]
    if faltantes:
        raise FileNotFoundError("Faltan archivos de entrada:\n" + "\n".join(faltantes))
    return archivos


def leer_poligonos(ruta):
    """Lee todos los polígonos, incluidos componentes separados e interiores."""
    faltantes = [
        str(ruta.with_suffix(ext))
        for ext in (".shp", ".shx", ".dbf", ".prj")
        if not ruta.with_suffix(ext).is_file()
    ]
    if faltantes:
        raise FileNotFoundError("Shapefile incompleto:\n" + "\n".join(faltantes))
    estados = gpd.read_file(ruta)
    if estados.empty or estados.crs is None:
        raise ValueError("El shapefile está vacío o no tiene un CRS definido.")
    if estados.geometry.isna().any() or estados.geometry.is_empty.any():
        raise ValueError("El shapefile contiene geometrías vacías o nulas.")
    if not estados.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        raise ValueError("El shapefile debe contener únicamente polígonos.")
    if not estados.geometry.is_valid.all():
        raise ValueError("El shapefile contiene geometrías inválidas; corrígelas antes del recorte.")
    estados = estados.to_crs(CRS)
    limites = tuple(estados.total_bounds)
    if not np.isfinite(limites).all() or not estados.geometry.is_valid.all():
        raise ValueError("El shapefile no produce geometrías válidas en WGS84.")
    print(f"Shapefile: {len(estados)} geometrías poligonales; CRS de trabajo: {CRS}.")
    return [mapping(geom) for geom in estados.geometry], limites


def validar_tiempo(datos, inicio, fin, etiqueta):
    """Detecta duplicados y meses ausentes, ordena y conserva fechas originales."""
    if "time" not in datos.coords or datos.time.dims != ("time",):
        raise ValueError(f"{etiqueta}: falta una coordenada time unidimensional.")
    if bool(datos.time.isnull().any()):
        raise ValueError(f"{etiqueta}: hay fechas nulas o inválidas.")
    indice = datos.time.to_index()
    if indice.has_duplicates:
        raise ValueError(f"{etiqueta}: hay fechas duplicadas; no se eliminarán automáticamente.")
    if not indice.is_monotonic_increasing:
        print(f"AVISO: {etiqueta}: se ordenaron las fechas cronológicamente.")
        datos = datos.sortby("time")
    try:
        meses = datos.time.dt.year.values * 12 + datos.time.dt.month.values - 1
    except (AttributeError, TypeError, ValueError):
        raise ValueError(f"{etiqueta}: time no contiene fechas mensuales decodificadas.") from None
    esperados = np.arange(inicio * 12, (fin + 1) * 12)
    if not np.array_equal(meses, esperados):
        raise ValueError(
            f"{etiqueta}: se esperaban {len(esperados)} registros, uno por mes, "
            f"de enero de {inicio} a diciembre de {fin}; se encontraron {len(meses)}. "
            "Hay meses ausentes, repetidos o fuera del período esperado."
        )
    return datos


def comprobar_eje(datos, nombre, etiqueta):
    """Admite ejes ascendentes o descendentes; exige grilla regular y finita."""
    if nombre not in datos.coords or datos[nombre].dims != (nombre,):
        raise ValueError(f"{etiqueta}: {nombre} debe ser una coordenada unidimensional.")
    valores = datos[nombre].values
    if not np.issubdtype(valores.dtype, np.number):
        raise ValueError(f"{etiqueta}: {nombre} no es una coordenada numérica.")
    if len(valores) < 2 or not np.isfinite(valores).all():
        raise ValueError(f"{etiqueta}: el eje {nombre} está incompleto o tiene valores inválidos.")
    diferencias = np.diff(valores.astype("float64"))
    if not ((diferencias > 0).all() or (diferencias < 0).all()):
        raise ValueError(f"{etiqueta}: {nombre} tiene duplicados o no es monótono.")
    paso = abs(float(valores[-1]) - float(valores[0])) / (len(valores) - 1)
    precision = np.finfo(valores.dtype).eps if valores.dtype.kind == "f" else 0
    tolerancia = max(1e-8, 4 * precision * float(np.max(np.abs(valores))))
    if not np.allclose(np.abs(diferencias), paso, rtol=1e-5, atol=tolerancia):
        raise ValueError(f"{etiqueta}: {nombre} no forma una grilla regular.")
    # La tolerancia anterior solo comprueba regularidad. La comparación
    # entre archivos y variables, más abajo, sí es EXACTA (array_equal).
    return valores, paso


def recortar_archivo(ruta, variable, anio, geometrias, limites):
    """Carga exclusivamente la ventana regional y aplica la máscara poligonal."""
    try:
        with xr.open_dataset(
            ruta, engine="netcdf4", decode_cf=True,
            mask_and_scale=True, decode_coords="all", cache=False
        ) as origen:
            if variable not in origen.data_vars:
                raise ValueError(f"No contiene la variable '{variable}'.")
            datos = origen[variable]
            if set(datos.dims) != {"time", "lat", "lon"}:
                raise ValueError(f"Dimensiones de '{variable}': {datos.dims}; se requieren time, lat, lon.")
            datos = validar_tiempo(datos, anio, anio, ruta.name)
            lat, dy = comprobar_eje(datos, "lat", ruta.name)
            lon, dx = comprobar_eje(datos, "lon", ruta.name)
            if lat.min() < -90 or lat.max() > 90:
                raise ValueError("La latitud no está expresada en grados geográficos.")
            if lon.min() >= 0 and lon.max() <= 360:
                # Admite también longitudes 0..360; no modifica las de -180..180.
                lon = np.where(lon >= 180, lon - 360, lon)
            if lon.min() < -180 or lon.max() > 180 or np.unique(lon).size != lon.size:
                raise ValueError("Las longitudes no son válidas o tienen duplicados tras normalizarlas.")
            if datos.rio.crs is not None and datos.rio.crs != CRS:
                raise ValueError(f"El NetCDF declara el CRS {datos.rio.crs}; se esperaba WGS84.")

            xmin, ymin, xmax, ymax = limites
            # Un margen de una celda evita perder bordes por redondeo.
            iy = np.flatnonzero((lat >= ymin - dy) & (lat <= ymax + dy))
            ix = np.flatnonzero((lon >= xmin - dx) & (lon <= xmax + dx))
            if iy.size < 2 or ix.size < 2:
                raise ValueError("No hay suficiente superposición entre la grilla y el shapefile.")
            sy = slice(int(iy.min()), int(iy.max()) + 1)
            sx = slice(int(ix.min()), int(ix.max()) + 1)
            # load ocurre DESPUÉS del isel: nunca se carga el raster global.
            regional = datos.isel(lat=sy, lon=sx).load()
            regional = regional.assign_coords(lon=("lon", lon[sx]))

        regional = regional.transpose("time", "lat", "lon")
        regional = regional.sortby("lat", ascending=False).sortby("lon")
        regional = regional.reset_coords(drop=True).astype("float32")
        # Los valores ya se decodificaron. Se retira el empaquetado original
        # para impedir una segunda aplicación de scale_factor/add_offset.
        regional.encoding = {}
        for clave in (
            "_FillValue", "missing_value", "scale_factor", "add_offset", "_Unsigned",
            "valid_min", "valid_max", "valid_range", "grid_mapping", "coordinates",
        ):
            regional.attrs.pop(clave, None)
        regional = regional.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
        regional = regional.rio.write_crs(CRS).rio.write_nodata(np.nan)
        regional = regional.rio.write_coordinate_system()

        # RECORTE REAL: utiliza todos los polígonos y conserva sus huecos.
        # drop=True reduce filas/columnas externas; dentro de la grilla
        # resultante, las celdas fuera del polígono quedan como NaN.
        return regional.rio.clip(
            geometrias, crs=CRS, drop=True, all_touched=False, from_disk=False
        )
    except Exception as exc:
        raise ValueError(f"Error al procesar {ruta}: {exc}") from None


def exigir_misma_grilla(a, b, etiqueta):
    for eje in ("lat", "lon"):
        if not np.array_equal(a[eje].values, b[eje].values):
            raise ValueError(
                f"{etiqueta}: las coordenadas {eje} NO coinciden exactamente. "
                "Se detiene el proceso; no se interpolarán ni alinearán silenciosamente."
            )


def unificar_variable(archivos, variable, geometrias, limites):
    partes = []
    print(f"\nProcesando {variable.upper()}: {len(archivos)} archivos...")
    for numero, (anio, ruta) in enumerate(
        zip(range(ANIO_INICIAL, ANIO_FINAL + 1), archivos), start=1
    ):
        parte = recortar_archivo(ruta, variable, anio, geometrias, limites)
        if partes:
            exigir_misma_grilla(partes[0], parte, ruta.name)
            if partes[0].attrs.get("units") != parte.attrs.get("units"):
                raise ValueError(f"{ruta.name}: las unidades difieren de las del primer año.")
        partes.append(parte)
        print(f"  [{numero:02d}/{len(archivos)}] {ruta.name}: 12 meses recortados.")

    # join='exact' impide que xarray forme una unión espacial con celdas nuevas.
    unido = xr.concat(partes, dim="time", coords="minimal", compat="equals", join="exact")
    partes.clear()
    unido = validar_tiempo(unido, ANIO_INICIAL, ANIO_FINAL, variable.upper())
    resultado = unido.to_dataset(name=variable)
    resultado.attrs = {
        "title": f"TerraClimate {variable}: noroeste de México, {ANIO_INICIAL}-{ANIO_FINAL}",
        "source": "TerraClimate; archivos anuales originales",
        "region": "Baja California, Baja California Sur, Sonora y Sinaloa",
        "clip_shapefile": str(SHAPEFILE),
        "clip_method": "rioxarray.rio.clip; all_touched=False; drop=True",
        "history": f"{datetime.now(timezone.utc).isoformat()}: unión mensual y recorte poligonal",
    }
    print(
        f"{variable.upper()}: {resultado.sizes['time']} pasos de tiempo; "
        f"{resultado.sizes['lat']} filas x {resultado.sizes['lon']} columnas."
    )
    print(
        f"  {resultado.time.values[0]} hasta {resultado.time.values[-1]}\n"
        "  Fechas duplicadas: NO; orden cronológico: SÍ; meses completos: SÍ."
    )
    return resultado


def comprobar_ppt_pet(ppt, pet):
    """Compara tamaños, cada coordenada y cada fecha, sin tolerancia numérica."""
    exigir_misma_grilla(ppt, pet, "Comparación PPT/PET")
    if ppt.sizes["time"] != pet.sizes["time"]:
        raise ValueError("PPT y PET tienen diferente número de pasos de tiempo.")
    if not np.array_equal(ppt.time.values, pet.time.values):
        raise ValueError("PPT y PET tienen fechas distintas, aunque puedan tener igual cantidad de meses.")
    print("\nVERIFICACIÓN PPT / PET:")
    print("  Coordenadas lat exactamente iguales: SÍ.")
    print("  Coordenadas lon exactamente iguales: SÍ.")
    print(f"  Igual número de pasos de tiempo: SÍ ({ppt.sizes['time']} en cada dataset).")
    print("  Fechas exactamente iguales: SÍ.")


def guardar_netcdf(dataset, variable, destino):
    """Escribe un temporal completo antes de reemplazar el resultado previo."""
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
    print(f"Guardado: {destino}")


def main():
    archivos_ppt = archivos_esperados(CARPETA_PPT, "ppt")
    archivos_pet = archivos_esperados(CARPETA_PET, "pet")
    geometrias, limites = leer_poligonos(SHAPEFILE)
    ppt = unificar_variable(archivos_ppt, "ppt", geometrias, limites)
    pet = unificar_variable(archivos_pet, "pet", geometrias, limites)
    # No se guarda ningún resultado hasta verificar los dos datasets.
    comprobar_ppt_pet(ppt, pet)
    print("\nGuardando NetCDF con NaN/NoData fuera de los polígonos...")
    guardar_netcdf(ppt, "ppt", SALIDA_PPT)
    guardar_netcdf(pet, "pet", SALIDA_PET)
    print("\nProceso completado correctamente.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
