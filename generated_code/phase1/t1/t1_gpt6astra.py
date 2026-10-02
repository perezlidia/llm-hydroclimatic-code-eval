"""Une y recorta la precipitación mensual TerraClimate de 1981 a 2025.

Instalación: python -m pip install xarray "dask[array]" netCDF4
Ejecución:   python unir_recortar_terraclimate.py
"""

from pathlib import Path

import xarray as xr
from dask.diagnostics import ProgressBar


CARPETA = Path("D:/2027-ARTICULOS/CAPITULO-SOFTWARE/ppt/")
ANIO_INICIAL, ANIO_FINAL = 1981, 2025
LAT_MIN, LAT_MAX = 22.5, 32.7
LON_MIN, LON_MAX = -117.2, -105.0


def recortar(ds):
    """Recorta antes de concatenar, sin cargar la precipitación global."""
    if "ppt" not in ds or set(ds["ppt"].dims) != {"time", "lat", "lon"}:
        raise ValueError("Se requiere 'ppt' con dimensiones time, lat y lon.")
    for nombre in ("time", "lat", "lon"):
        if nombre not in ds.coords or ds[nombre].dims != (nombre,):
            raise ValueError(f"La coordenada {nombre} debe ser unidimensional.")

    variables = ["ppt"]
    referencia_crs = ds.ppt.attrs.get("grid_mapping")
    if referencia_crs in ds.variables:
        variables.append(referencia_crs)
    ds = ds[variables]

    # Admite también longitudes entre 0 y 360 grados.
    if float(ds.lon.max()) > 180:
        atributos_lon = ds.lon.attrs.copy()
        ds = ds.assign_coords(lon=xr.where(ds.lon > 180, ds.lon - 360, ds.lon))
        ds.lon.attrs = atributos_lon

    # Las máscaras funcionan con coordenadas ascendentes o descendentes.
    ds = ds.isel(
        lat=((ds.lat >= LAT_MIN) & (ds.lat <= LAT_MAX)).values,
        lon=((ds.lon >= LON_MIN) & (ds.lon <= LON_MAX)).values,
    )
    if ds.sizes["lat"] == 0 or ds.sizes["lon"] == 0:
        raise ValueError("El recorte espacial no contiene celdas.")

    # Unifica la orientación espacial de todos los archivos.
    return ds.sortby("lat").sortby("lon")


def main(carpeta=CARPETA):
    carpeta = Path(carpeta)
    archivos = [
        carpeta / f"TerraClimate_ppt_{anio}.nc"
        for anio in range(ANIO_INICIAL, ANIO_FINAL + 1)
    ]
    faltantes = [str(archivo) for archivo in archivos if not archivo.is_file()]
    if faltantes:
        raise FileNotFoundError("Faltan archivos:\n" + "\n".join(faltantes))

    salida = carpeta / "TerraClimate_ppt_1981_2025_recortado.nc"
    print(f"Abriendo {len(archivos)} archivos...")

    # Dask mantiene los datos en bloques; no carga los 45 archivos completos.
    # 'nested' concatena todos los registros, incluidos posibles duplicados.
    with xr.open_mfdataset(
        archivos,
        engine="netcdf4",
        combine="nested",
        concat_dim="time",
        preprocess=recortar,
        chunks={"time": 12, "lat": 256, "lon": 256},
        decode_times=True,
        mask_and_scale=True,
        data_vars="minimal",
        coords="minimal",
        compat="no_conflicts",
        join="exact",  # Exige la misma cuadrícula después del recorte.
        combine_attrs="drop_conflicts",
    ) as conjunto:
        if bool(conjunto.time.isnull().any()):
            raise ValueError("Hay fechas vacías o inválidas.")

        fechas = conjunto.indexes["time"]
        if fechas.has_duplicates:
            duplicadas = fechas[fechas.duplicated()].unique()
            raise ValueError(f"Hay fechas duplicadas: {list(duplicadas[:10])}")

        # Ordena por las fechas reales, independientemente del orden original.
        ds = conjunto.sortby("time")
        if not ds.indexes["time"].is_monotonic_increasing:
            raise ValueError("No fue posible ordenar cronológicamente la serie.")

        # Comprueba un registro por mes; no exige que la fecha sea el día 1.
        meses = ds.time.dt.strftime("%Y-%m").values.tolist()
        esperados = [
            f"{anio}-{mes:02d}"
            for anio in range(ANIO_INICIAL, ANIO_FINAL + 1)
            for mes in range(1, 13)
        ]
        if meses != esperados:
            raise ValueError(
                "La serie debe contener exactamente 540 meses, de 1981-01 "
                "a 2025-12. Hay meses faltantes, repetidos o fuera del período."
            )

        ds["ppt"] = ds.ppt.transpose("time", "lat", "lon")
        # Recalcula la codificación de salida a partir de los datos decodificados.
        ds = ds.drop_encoding()
        ds.attrs.update(
            geospatial_lat_min=float(ds.lat.min()),
            geospatial_lat_max=float(ds.lat.max()),
            geospatial_lon_min=float(ds.lon.min()),
            geospatial_lon_max=float(ds.lon.max()),
            time_coverage_start=str(ds.time.values[0]),
            time_coverage_end=str(ds.time.values[-1]),
        )
        ds.attrs.pop("time_coverage_duration", None)

        print(f"Pasos de tiempo: {ds.sizes['time']}")
        print(f"Dimensiones espaciales: lat={ds.sizes['lat']}, lon={ds.sizes['lon']}")
        print("Fechas únicas, ordenadas y sin meses faltantes.")
        print(f"Guardando: {salida}")

        # mode='w' reemplaza este archivo de salida si ya existe.
        with ProgressBar():
            ds.to_netcdf(
                salida,
                mode="w",
                engine="netcdf4",
                format="NETCDF4",
                encoding={"ppt": {"zlib": True, "complevel": 4}},
            )

    print(f"Archivo guardado: {salida}")


if __name__ == "__main__":
    main()
