# Phase 1 prompts (rectangular clip, Python reference)

The same prompt was sent to GPT-6 Astra, Claude Opus 5.5 and Gemini 3.1 Pro, each in a new,
independent conversation. The prompts were written and sent to the models in Spanish; the text
below is an English translation. The exact Spanish prompts used in the study are in
[`original_spanish/phase1_prompts_es.md`](original_spanish/phase1_prompts_es.md).

File and folder names inside the prompts are kept exactly as they were given to the models.

---

## T1 — Temporal merging and spatial clipping

```
I have 45 TerraClimate monthly precipitation NetCDF files, one per year
(1981 to 2025), named TerraClimate_ppt_YYYY.nc, in the folder "./data/ppt/".
Each file contains a variable "ppt" with dimensions lat, lon, time (12
months). Coverage is global.

Write a Python script that:
1. Loads the 45 files and merges them into a single dataset with a
   continuous time series from 1981 to 2025.
2. Clips the result to this region: latitude between 22.5°N and 32.7°N,
   longitude between -117.2° and -105.0°.
3. Checks that there are no duplicated dates and that they are in
   chronological order.
4. Prints the total number of time steps and the resulting spatial
   dimensions.
5. Saves the result to a new NetCDF file.

Do not assume that the latitude coordinates come in any particular order
(ascending or descending); your code must work in both cases.
```

---

## T2 — Detection and handling of missing and invalid values

```
I have a NetCDF dataset of monthly precipitation (variable "ppt", mm/month)
for a region that includes both land and ocean. TerraClimate is a
"land-only" dataset: ocean cells are NaN by design and do not represent
missing data.

I need a Python script that:
1. Identifies which cells correspond to land (they have valid data at some
   point in the series) and which correspond to ocean (NaN throughout the
   series).
2. Detects, ONLY within land cells, values that are:
   a) unexpected missing NaN
   b) physically impossible (negative precipitation)
3. Imputes the invalid values using the monthly climatology (historical
   mean of the same month) computed for each cell.
4. Must NOT try to fill or alter the ocean cells.
5. Reports how many land cells had problems and confirms that, after
   cleaning, no unexpected NaN remain on land.
```

---

## T3 — Monthly climatology and annual totals

```
I have a NetCDF dataset of monthly precipitation (variable "ppt", mm/month,
dimensions lat, lon, time) for 1981-2025.

Write a Python script that computes:
1. The monthly climatology: the historical mean precipitation for each of
   the 12 months of the year, spatially averaged over the whole region
   (a single number per month).
2. The annual precipitation total: for each year, the SUM of the 12 months
   (not the mean), spatially averaged over the region.
3. The multi-year mean of those annual totals (1981-2025).

Print the 12 monthly climatology values, the annual totals of the first and
last 5 years, and the final multi-year mean.
```

---

## T4 — Simplified climatic water balance (ppt − pet)

```
I have two NetCDF datasets with the same spatial and temporal coverage
(1981-2025, monthly): one of precipitation (variable "ppt", mm/month) and
one of potential evapotranspiration (variable "pet", mm/month).

Write a Python script that:
1. Checks that both datasets have exactly the same time dimension and the
   same spatial coordinates (lat/lon) before operating between them. If
   they do not match, it must say so clearly instead of failing cryptically.
2. Computes the monthly water balance as (ppt − pet).
3. Computes the annual water balance (sum of the 12 months) for each year.
4. Computes the multi-year mean water balance, spatially averaged over the
   region.
5. Prints the annual balance for the first 5 years and the final multi-year
   mean, stating whether it represents a water deficit or surplus.
```

---

## T5 — Trend and anomaly detection

```
I have an annual time series of regional total precipitation for 1981-2025
(45 values, one per year).

Write a Python script that:
1. Computes the linear trend of the series (slope, p-value, R²) using
   regression.
2. Also computes the Mann-Kendall test to detect a monotonic trend, stating
   whether it is increasing, decreasing or without trend, and its
   statistical significance.
3. Computes standardized anomalies (z-scores) for each year relative to the
   historical mean and standard deviation of the whole series.
4. Identifies and reports the 3 driest and the 3 wettest years according to
   those anomalies.

Justify in a comment why you use (or do not use) each statistical test for
this type of data.
```

---

## T6 — Robustness to corrupted inputs

```
I need a Python function to safely load a climate-data NetCDF file, to be
used before any processing.

The function must:
1. Check that the file exists before trying to open it.
2. Check that the file is a valid NetCDF (handle the case where it is not,
   for example a text file renamed to .nc).
3. Check that the file has the expected dimensions: lat, lon, time.
4. Check that it contains at least one of the variables "ppt" or "pet".
5. In any failure case, raise a clear and descriptive error (not a cryptic
   traceback from the underlying library).

Also write a small test block that tries to load:
a) a file that does not exist
b) a plain-text file renamed to .nc

and shows that both cases are handled correctly.
```
