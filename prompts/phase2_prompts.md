# Phase 2 prompts (polygon clip, ArcMap reference)

The same prompt was sent to GPT-6 Astra, Claude Opus 5.5 and Gemini 3.1 Pro, each in a new,
independent conversation. The prompts were written and sent to the models in Spanish; the text
below is an English translation. The exact Spanish prompts used in the study are in
[`original_spanish/phase2_prompts_es.md`](original_spanish/phase2_prompts_es.md).

Compared with Phase 1, these prompts (i) refer to the state-polygon clip, (ii) state explicitly how
the `sum()` function must treat all-NaN cells (T3–T5), and (iii) require the function and its tests
to be delivered in a single, self-executing file (T6).

File and folder names inside the prompts (for example `ppt_noroeste_shp_1981_2025.nc` or
`Noroeste.shp`) are kept exactly as they were given to the models, because the generated scripts
use them. Their published English equivalents are listed in `../data/README.md`.

---

## T1 — Temporal merging and polygon clipping

```
I have 45 TerraClimate monthly precipitation NetCDF files and 45 potential
evapotranspiration files, one per year (1981 to 2025), named
TerraClimate_ppt_YYYY.nc in the folder "D:/2027-ARTICULOS/CAPITULO-SOFTWARE/ppt/"
and TerraClimate_pet_YYYY.nc in "D:/2027-ARTICULOS/CAPITULO-SOFTWARE/pet/".
Each file contains a variable "ppt" or "pet" respectively, with dimensions
lat, lon, time (12 months). Coverage is global.

I also have a shapefile named "Noroeste.shp" (with its associated .shx,
.dbf and .prj files) in the folder "D:/2027-ARTICULOS/CAPITULO-SOFTWARE/Noroeste/",
containing the polygons of 4 Mexican states (Baja California, Baja
California Sur, Sonora and Sinaloa), in WGS84 geographic coordinates.

Write a Python script that:
1. Loads the 45 precipitation files and merges them into a single dataset
   with a continuous time series from 1981 to 2025.
2. Loads the 45 potential evapotranspiration files and merges them in the
   same way.
3. Clips BOTH datasets using the ACTUAL POLYGON of the shapefile
   "Noroeste.shp" (not a rectangle/bounding box) — cells outside the
   polygon must be NaN/NoData, not merely excluded by a rectangular
   lat/lon range.
4. Checks that there are no duplicated dates and that dates are in
   chronological order in both datasets, and that both have EXACTLY the
   same spatial coordinates (lat/lon) and the same number of time steps
   after clipping — report this explicitly.
5. Prints the total number of time steps and the resulting spatial
   dimensions (rows x columns of the clipped grid) of each dataset.
6. Saves each result to a new NetCDF file, named
   "ppt_noroeste_shp_1981_2025.nc" and "pet_noroeste_shp_1981_2025.nc"
   (with the "_shp" suffix to distinguish them from an earlier version
   clipped with a simple rectangle).

Use geopandas to read the shapefile and rioxarray (or an equivalent
library) to clip the NetCDF by the exact geometry. Do not assume that the
latitude coordinates come in any particular order.
```

---

## T2 — Detection and handling of missing and invalid values

```
I have a NetCDF file of monthly precipitation (variable "ppt", mm/month,
dimensions time, lat, lon) already clipped to a region that includes land
and cells outside the study polygon (marked as NaN/NoData). The file is
named "ppt_noroeste_shp_CORRUPTO_prueba_T2.nc" and is in the same folder as
the script.

Cells outside the polygon (NaN at ALL time steps) represent areas outside
the study area, not actual missing data.

I need a Python script that:
1. Identifies which cells belong to the study area (they have valid data at
   some point in the series) and which are outside the polygon (NaN
   throughout the series).
2. Detects, ONLY within the study-area cells, values that are:
   a) unexpected missing NaN
   b) physically impossible (negative precipitation)
3. Imputes the invalid values using the monthly climatology (historical
   mean of the same month) computed for each cell.
4. Must NOT try to fill or alter the cells outside the study polygon.
5. Reports how many study-area cells had problems and confirms that, after
   cleaning, no unexpected NaN remain within the study area.
```

---

## T3 — Monthly climatology and annual totals

```
I have a NetCDF file of monthly precipitation (variable "ppt", mm/month,
dimensions time, lat, lon) already clipped to the actual polygon of 4
states of northwestern Mexico, for 1981-2025. The file is named
"ppt_noroeste_shp_1981_2025.nc" and is in the same folder as the script.
Cells outside the polygon are NaN/NoData.

Write a Python script that computes:
1. The monthly climatology: the historical mean precipitation for each of
   the 12 months of the year, spatially averaged over the cells inside the
   polygon (excluding cells outside the study area).
2. The annual precipitation total: for each year, the SUM of the 12 months
   (not the mean), spatially averaged over the study area.
3. The multi-year mean of those annual totals (1981-2025).

IMPORTANT: when summing the 12 months of each year, make sure that cells
outside the study polygon yield NaN (not zero) in the total — check how
the sum function you use behaves with NaN values before applying it.

Print the 12 monthly climatology values, the annual totals of the first and
last 5 years, and the final multi-year mean.
```

---

## T4 — Simplified climatic water balance (ppt − pet)

```
I have two NetCDF files already clipped to the actual polygon of 4 states
of northwestern Mexico, with the same spatial and temporal coverage
(1981-2025, monthly): one of precipitation ("ppt_noroeste_shp_1981_2025.nc",
variable "ppt", mm/month) and one of potential evapotranspiration
("pet_noroeste_shp_1981_2025.nc", variable "pet", mm/month). Cells outside
the polygon are NaN/NoData in both files.

Write a Python script that:
1. Checks that both datasets have exactly the same time dimension and the
   same spatial coordinates (lat/lon) before operating between them.
2. Computes the monthly water balance as (ppt − pet).
3. Computes the annual water balance (sum of the 12 months) for each year,
   making sure that cells outside the polygon yield NaN (not zero) in the
   total.
4. Computes the multi-year mean water balance, spatially averaged over the
   study area.
5. Prints the annual balance for the first 5 years and the final multi-year
   mean, stating whether it represents a water deficit or surplus.
```

---

## T5 — Trend and anomaly detection

```
I have a NetCDF file of monthly precipitation (variable "ppt", mm/month,
dimensions time, lat, lon) already clipped to the actual polygon of 4
states of northwestern Mexico, for 1981-2025. The file is named
"ppt_noroeste_shp_1981_2025.nc" and is in the same folder as the script.
Cells outside the polygon are NaN/NoData.

Write a Python script that:
1. Computes the regional annual total series (sum of the 12 months of each
   year, spatially averaged over the study area inside the polygon). Make
   sure that cells outside the polygon do not contaminate the result
   (check how the sum function behaves with NaN).
2. Computes the linear trend of that series (slope, p-value, R²).
3. Also computes the Mann-Kendall test to detect a monotonic trend, stating
   whether it is increasing, decreasing or without trend, and its
   statistical significance.
4. Computes standardized anomalies (z-scores) for each year relative to the
   historical mean and standard deviation of the whole series.
5. Identifies and reports the 3 driest and the 3 wettest years according to
   those anomalies.

Justify in a comment why you use (or do not use) each statistical test for
this type of data.
```

---

## T6 — Robustness to corrupted inputs

```
I need a Python function to safely load a climate-data NetCDF file, to be
used before any processing. The actual files this function will be used
with are named, for example, "ppt_noroeste_shp_1981_2025.nc" and
"pet_noroeste_shp_1981_2025.nc" (precipitation and potential
evapotranspiration, clipped to a study polygon, with cells outside the
polygon as NaN/NoData).

The function must:
1. Check that the file exists before trying to open it.
2. Check that the file is a valid NetCDF (handle the case where it is not,
   for example a text file renamed to .nc).
3. Check that the file has the expected dimensions: lat, lon, time.
4. Check that it contains at least one of the variables "ppt" or "pet".
5. In any failure case, raise a clear and descriptive error (not a cryptic
   traceback from the underlying library).

Also write a small test block, in the SAME file (running automatically when
the script is executed, with no additional manual steps), that tries to
load:
a) a file that does not exist
b) a plain-text file renamed to .nc

and shows that both cases are handled correctly.
```
