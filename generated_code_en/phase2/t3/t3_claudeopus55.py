# English translation of the original Spanish delivery (comments and messages only; code unchanged). Original: generated_code/phase2/t3/t3_claudeopus55.py
"""
Monthly climatology and annual accumulated precipitation (1981-2025)
Study area: 4 states of Northwest Mexico (polygon clip).

Input  : ppt_noroeste_shp_1981_2025.nc  (variable 'ppt', mm/month; dims time, lat, lon)
Output : console printout of
          1) monthly climatology (12 values, mm/month)
          2) annual totals (first and last 5 years, mm/year)
          3) multi-year mean of the annual totals (mm/year)

Note on NaN and the sum:
    In xarray (and numpy.nansum), .sum() uses skipna=True by default, and the sum
    of a series made up ONLY of NaN returns 0, not NaN. That would turn
    cells outside the polygon into zeros and bias the spatial mean
    downward. To avoid it, min_count=12 is used: if a cell does not have all 12
    valid months of the year, its annual total stays NaN.
"""

from pathlib import Path

import numpy as np
import xarray as xr

ARCHIVO = Path(__file__).resolve().parent / "ppt_noroeste_shp_1981_2025.nc"
VARIABLE = "ppt"
ANIO_INI, ANIO_FIN = 1981, 2025

MESES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def nombre_coord(da, candidatos):
    """Returns the actual name of a coordinate (lat/latitude, lon/longitude...)."""
    for c in candidatos:
        if c in da.dims or c in da.coords:
            return c
    raise KeyError(f"None of the coordinates {candidatos} were found")


def promedio_espacial(da, lat, lon):
    """
    Spatial mean weighted by cos(latitude), ignoring NaN.
    NaN cells (outside the polygon) contribute neither to the numerator nor
    to the sum of weights.
    """
    pesos = np.cos(np.deg2rad(da[lat]))
    pesos.name = "pesos"
    return da.weighted(pesos).mean(dim=(lat, lon), skipna=True)


def main():
    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------
    ds = xr.open_dataset(ARCHIVO)
    ppt = ds[VARIABLE]

    # Make sure NoData values declared as _FillValue are already NaN
    # (xarray does this when decoding; this covers cases with sentinel values)
    ppt = ppt.where(ppt > -9000)

    lat = nombre_coord(ppt, ["lat", "latitude", "y"])
    lon = nombre_coord(ppt, ["lon", "longitude", "x"])

    ppt = ppt.sel(time=slice(f"{ANIO_INI}-01-01", f"{ANIO_FIN}-12-31"))

    # Study-area mask: cells with at least one valid value
    mascara = ppt.notnull().any(dim="time")
    n_celdas_poligono = int(mascara.sum())
    n_celdas_total = mascara.size
    print(f"Cells inside the polygon: {n_celdas_poligono} of {n_celdas_total}")

    # ------------------------------------------------------------------
    # Demonstration of the behavior of .sum() with NaN
    # ------------------------------------------------------------------
    prueba = xr.DataArray([np.nan] * 12, dims="time")
    print("\nCheck of the behavior of the sum with NaN:")
    print(f"  default sum() over 12 NaN      -> {float(prueba.sum()):.1f}  (it gives 0!)")
    print(f"  sum(min_count=12) over 12 NaN  -> {float(prueba.sum(min_count=12))}")

    # ------------------------------------------------------------------
    # 1) Monthly climatology
    # ------------------------------------------------------------------
    # Mean per month in each cell (mean ignores NaN; cells outside
    # the polygon are NaN at all times and stay NaN)
    clim_celda = ppt.groupby("time.month").mean(dim="time", skipna=True)
    clim_mensual = promedio_espacial(clim_celda, lat, lon)

    print("\n1) Monthly climatology (mm/month), spatial mean "
          f"{ANIO_INI}-{ANIO_FIN}:")
    for m, v in zip(clim_mensual["month"].values, clim_mensual.values):
        print(f"   {MESES[m - 1]}: {v:8.2f}")
    print(f"   Sum of the monthly climatology: {float(clim_mensual.sum()):.2f} mm/year")

    # ------------------------------------------------------------------
    # 2) Annual total (SUM of 12 months)
    # ------------------------------------------------------------------
    # Check that each year has all 12 months on the time axis
    meses_por_anio = ppt["time"].groupby("time.year").count()
    anios_incompletos = meses_por_anio.where(meses_por_anio < 12, drop=True)
    if anios_incompletos.size > 0:
        print("\nWARNING: years with fewer than 12 months (excluded from the annual total):")
        for a, n in zip(anios_incompletos["year"].values, anios_incompletos.values):
            print(f"   {a}: {int(n)} months")

    # min_count=12 -> NaN if the cell does not have all 12 valid months
    acum_celda = ppt.groupby("time.year").sum(dim="time", skipna=True, min_count=12)
    anios_completos = meses_por_anio["year"].where(meses_por_anio == 12, drop=True)
    acum_celda = acum_celda.sel(year=anios_completos.values.astype(int))

    # Check: outside the polygon there must be NaN, not zeros
    fuera = ~mascara
    ceros_fuera = int((acum_celda.where(fuera) == 0).sum())
    nan_fuera_ok = bool(acum_celda.where(fuera).isnull().all()) if int(fuera.sum()) else True
    print("\nCheck of the annual total:")
    print(f"   Zeros in cells outside the polygon: {ceros_fuera}")
    print(f"   All cells outside the polygon are NaN: {nan_fuera_ok}")
    assert ceros_fuera == 0 and nan_fuera_ok, \
        "There are cells outside the polygon with a non-NaN value in the annual total."

    acum_anual = promedio_espacial(acum_celda, lat, lon)

    print("\n2) Annual total (mm/year), spatial mean:")
    print("   First 5 years:")
    for a, v in zip(acum_anual["year"].values[:5], acum_anual.values[:5]):
        print(f"     {a}: {v:8.2f}")
    print("   Last 5 years:")
    for a, v in zip(acum_anual["year"].values[-5:], acum_anual.values[-5:]):
        print(f"     {a}: {v:8.2f}")

    # ------------------------------------------------------------------
    # 3) Multi-year mean of the annual totals
    # ------------------------------------------------------------------
    promedio_multianual = float(acum_anual.mean(skipna=True))
    a0, a1 = int(acum_anual["year"].min()), int(acum_anual["year"].max())
    print(f"\n3) Multi-year mean of the annual total ({a0}-{a1}, "
          f"{acum_anual.size} years): {promedio_multianual:.2f} mm/year")

    ds.close()


if __name__ == "__main__":
    main()
