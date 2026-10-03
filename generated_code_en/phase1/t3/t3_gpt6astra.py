# English translation of the original Spanish delivery (comments and messages only; code unchanged). Original: generated_code/phase1/t3/t3_gpt6astra.py
"""Monthly climatology and regional annual precipitation totals.

Installation: python -m pip install numpy xarray netCDF4
Execution:    python resumen_ppt_1981_2025.py

Place ppt_noroeste_1981_2025.nc in the same folder as this script.
The variable ppt contains monthly totals in mm/month.

The spatial average is arithmetic, without area weighting. NaN values are
excluded from the averages; cells that are always NaN never contribute zeros.
For each cell, an annual total requires all 12 valid months. If any is
missing, that cell is excluded from the spatial average for that year. Hence, with
missing values on land, the spatial coverage may vary between years.
The monthly climatology uses the available data for each month and cell.
The input file is not modified and no values are imputed.
"""

from pathlib import Path

import numpy as np
import xarray as xr


ARCHIVO = Path(__file__).resolve().parent / "ppt_noroeste_1981_2025.nc"
ANIO_INICIAL = 1981
ANIO_FINAL = 2025
MESES = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)


def calcular_resumen(ppt):
    """Returns climatology (month), totals (year) and multi-year mean."""
    if set(ppt.dims) != {"time", "lat", "lon"}:
        raise ValueError("The variable 'ppt' must have dimensions time, lat and lon.")

    # Selecting by year also supports dates placed at the end of the month.
    ppt = ppt.sortby("time").sel(
        time=slice(str(ANIO_INICIAL), str(ANIO_FINAL))
    )

    # Require exactly one record per month between January 1981 and December
    # 2025. This detects missing months and duplicate months, even if the
    # duplicate dates within a month have different days.
    claves = ppt.time.dt.year.values * 12 + ppt.time.dt.month.values - 1
    esperadas = np.arange(ANIO_INICIAL * 12, (ANIO_FINAL + 1) * 12)
    if not np.array_equal(claves, esperadas):
        raise ValueError(
            "540 records are required: one per month between 1981 and 2025. "
            "Check whether there are missing or duplicate months."
        )

    # 1. Historical mean of each month in each cell; then, spatial mean.
    climatologia = (
        ppt.groupby("time.month")
        .mean(dim="time", skipna=True, dtype=np.float64)
        .mean(dim=("lat", "lon"), skipna=True)
    )

    # 2. SUM of the 12 months in each cell; then, spatial mean.
    # min_count=12 avoids partial totals and keeps the ocean as NaN.
    acumulados_por_celda = ppt.groupby("time.year").sum(
        dim="time", skipna=True, min_count=12, dtype=np.float64
    )
    acumulados_anuales = acumulados_por_celda.mean(
        dim=("lat", "lon"), skipna=True
    )

    if not np.isfinite(climatologia.values).all():
        raise ValueError("A valid climatology was not obtained for the 12 months.")
    if not np.isfinite(acumulados_anuales.values).all():
        raise ValueError(
            "A valid regional total was not obtained for all years. "
            "Check that each year has cells with all 12 months valid."
        )

    # 3. Mean of the 45 regional annual totals, with equal weight per year.
    promedio_multianual = acumulados_anuales.mean(
        dim="year", skipna=False
    ).item()

    return climatologia, acumulados_anuales, promedio_multianual


def main():
    if not ARCHIVO.is_file():
        raise FileNotFoundError(f"File not found: {ARCHIVO}")

    # Decodes dates, _FillValue and scale_factor/add_offset from the NetCDF.
    with xr.open_dataset(ARCHIVO, decode_times=True, mask_and_scale=True) as ds:
        if "ppt" not in ds:
            raise ValueError("The file does not contain the variable 'ppt'.")
        climatologia, anuales, multianual = calcular_resumen(ds["ppt"])

    print("PERIOD: 1981-2025 (45 years; 540 months)")
    print("Spatial average: arithmetic over valid cells, without area weighting.")
    print("Each annual total per cell requires 12 valid months.")

    print("\nREGIONAL MONTHLY CLIMATOLOGY (mm/month)")
    for numero, nombre in enumerate(MESES, start=1):
        valor = climatologia.sel(month=numero).item()
        print(f"{nombre:<12}: {valor:10.2f} mm/month")

    for titulo, inicio, fin in (
        ("FIRST 5 YEARS", ANIO_INICIAL, ANIO_INICIAL + 4),
        ("LAST 5 YEARS", ANIO_FINAL - 4, ANIO_FINAL),
    ):
        print(f"\nREGIONAL ANNUAL TOTALS - {titulo} (mm/year)")
        for anio in range(inicio, fin + 1):
            valor = anuales.sel(year=anio).item()
            print(f"{anio}: {valor:10.2f} mm/year")

    print("\nMULTI-YEAR AVERAGE OF THE ANNUAL TOTALS (1981-2025)")
    print(f"{multianual:.2f} mm/year")


if __name__ == "__main__":
    main()
