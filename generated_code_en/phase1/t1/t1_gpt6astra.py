# English translation of the original Spanish delivery (comments and messages only; code unchanged). Original: generated_code/phase1/t1/t1_gpt6astra.py
"""Merges and crops TerraClimate monthly precipitation from 1981 to 2025.

Installation: python -m pip install xarray "dask[array]" netCDF4
Execution:    python unir_recortar_terraclimate.py
"""

from pathlib import Path

import xarray as xr
from dask.diagnostics import ProgressBar


CARPETA = Path("D:/2027-ARTICULOS/CAPITULO-SOFTWARE/ppt/")
ANIO_INICIAL, ANIO_FINAL = 1981, 2025
LAT_MIN, LAT_MAX = 22.5, 32.7
LON_MIN, LON_MAX = -117.2, -105.0


def recortar(ds):
    """Crops before concatenating, without loading the global precipitation."""
    if "ppt" not in ds or set(ds["ppt"].dims) != {"time", "lat", "lon"}:
        raise ValueError("'ppt' is required with dimensions time, lat and lon.")
    for nombre in ("time", "lat", "lon"):
        if nombre not in ds.coords or ds[nombre].dims != (nombre,):
            raise ValueError(f"The coordinate {nombre} must be one-dimensional.")

    variables = ["ppt"]
    referencia_crs = ds.ppt.attrs.get("grid_mapping")
    if referencia_crs in ds.variables:
        variables.append(referencia_crs)
    ds = ds[variables]

    # Also supports longitudes between 0 and 360 degrees.
    if float(ds.lon.max()) > 180:
        atributos_lon = ds.lon.attrs.copy()
        ds = ds.assign_coords(lon=xr.where(ds.lon > 180, ds.lon - 360, ds.lon))
        ds.lon.attrs = atributos_lon

    # The masks work with ascending or descending coordinates.
    ds = ds.isel(
        lat=((ds.lat >= LAT_MIN) & (ds.lat <= LAT_MAX)).values,
        lon=((ds.lon >= LON_MIN) & (ds.lon <= LON_MAX)).values,
    )
    if ds.sizes["lat"] == 0 or ds.sizes["lon"] == 0:
        raise ValueError("The spatial crop contains no cells.")

    # Unifies the spatial orientation of all files.
    return ds.sortby("lat").sortby("lon")


def main(carpeta=CARPETA):
    carpeta = Path(carpeta)
    archivos = [
        carpeta / f"TerraClimate_ppt_{anio}.nc"
        for anio in range(ANIO_INICIAL, ANIO_FINAL + 1)
    ]
    faltantes = [str(archivo) for archivo in archivos if not archivo.is_file()]
    if faltantes:
        raise FileNotFoundError("Missing files:\n" + "\n".join(faltantes))

    salida = carpeta / "TerraClimate_ppt_1981_2025_recortado.nc"
    print(f"Opening {len(archivos)} files...")

    # Dask keeps the data in chunks; it does not load the 45 full files.
    # 'nested' concatenates all records, including possible duplicates.
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
        join="exact",  # Requires the same grid after cropping.
        combine_attrs="drop_conflicts",
    ) as conjunto:
        if bool(conjunto.time.isnull().any()):
            raise ValueError("There are empty or invalid dates.")

        fechas = conjunto.indexes["time"]
        if fechas.has_duplicates:
            duplicadas = fechas[fechas.duplicated()].unique()
            raise ValueError(f"There are duplicate dates: {list(duplicadas[:10])}")

        # Sorts by the actual dates, regardless of the original order.
        ds = conjunto.sortby("time")
        if not ds.indexes["time"].is_monotonic_increasing:
            raise ValueError("It was not possible to sort the series chronologically.")

        # Checks for one record per month; does not require the date to be day 1.
        meses = ds.time.dt.strftime("%Y-%m").values.tolist()
        esperados = [
            f"{anio}-{mes:02d}"
            for anio in range(ANIO_INICIAL, ANIO_FINAL + 1)
            for mes in range(1, 13)
        ]
        if meses != esperados:
            raise ValueError(
                "The series must contain exactly 540 months, from 1981-01 "
                "to 2025-12. There are missing, repeated or out-of-period months."
            )

        ds["ppt"] = ds.ppt.transpose("time", "lat", "lon")
        # Recomputes the output encoding from the decoded data.
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

        print(f"Time steps: {ds.sizes['time']}")
        print(f"Spatial dimensions: lat={ds.sizes['lat']}, lon={ds.sizes['lon']}")
        print("Dates are unique, sorted and with no missing months.")
        print(f"Saving: {salida}")

        # mode='w' replaces this output file if it already exists.
        with ProgressBar():
            ds.to_netcdf(
                salida,
                mode="w",
                engine="netcdf4",
                format="NETCDF4",
                encoding={"ppt": {"zlib": True, "complevel": 4}},
            )

    print(f"File saved: {salida}")


if __name__ == "__main__":
    main()
