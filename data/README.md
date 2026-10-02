# Data

Because of their size, the data are not stored in the Git tree. The derived data are distributed as
`llm-hydroclimatic-code-eval_data.zip`, attached to release
[v1.0.0](https://github.com/perezlidia/llm-hydroclimatic-code-eval/releases/tag/v1.0.0) of this
repository; the source data can be downloaded from their providers.

## Source data

| Data | Source |
|---|---|
| TerraClimate monthly `ppt` and `pet`, 1981–2025 (annual files `TerraClimate_<var>_<year>.nc`) | Climatology Lab, https://www.climatologylab.org/terraclimate.html (Abatzoglou et al. 2018). Downloaded on 5 August 2026. |
| State boundaries of Baja California, Baja California Sur, Sonora and Sinaloa (published as `northwest_mexico_states.shp`, WGS 84, EPSG:4326) | Instituto Nacional de Estadística y Geografía (INEGI), https://www.inegi.org.mx. Downloaded on 5 August 2026. |

## Derived data used in the study (release v1.0.0 of this repository)

The data package `llm-hydroclimatic-code-eval_data.zip` contains:

| File | Description |
|---|---|
| `phase1_ppt_rectangular_clip_1981_2025.nc` | Phase 1 rectangular clip (common input for Phase 1, T2–T6) |
| `phase1_pet_rectangular_clip_1981_2025.nc`, `phase1_t2_corrupted_input_ppt.nc` | Phase 1 `pet` clip (T4) and corrupted Phase 1 T2 input; distributed in the second release asset `llm-hydroclimatic-code-eval_data_phase1_supplement.zip` |
| `phase2_ppt_polygon_clip_1981_2025.nc`, `phase2_pet_polygon_clip_1981_2025.nc` | Phase 2 polygon clip (common input for Phase 2, T2–T6; identical for the three models) |
| `phase2_t2_corrupted_input_ppt.nc` | Corrupted Phase 2 T2 input: 99 cells negative in all 540 months and 394 cells NaN in all 540 months |
| `T1_<model>_ppt.nc`, `T1_<model>_pet.nc` | Phase 2 T1 clips produced by each model |
| `arcmap/t4_water_balance_arcmap.tif` | ArcMap reference, mean multi-year water balance (ppt − pet) |
| `arcmap/t1_ppt_1981_clip_arcmap.tif` | ArcMap reference clip (1981, 12 bands), used for the T1 comparison |
| `study_area/northwest_mexico_states.shp` | State polygons used for the Phase 2 clip (INEGI) |

## Original file names

The prompts, and therefore the model scripts, refer to the input files by their original Spanish
names. `verification/prepare_workdir.py` restores these names in the working directory, so that
the unmodified model scripts find their inputs:

| Published name | Name expected by the model scripts |
|---|---|
| `phase1_ppt_rectangular_clip_1981_2025.nc` | `ppt_noroeste_1981_2025.nc` |
| `phase1_pet_rectangular_clip_1981_2025.nc` | `pet_noroeste_1981_2025.nc` |
| `phase1_t2_corrupted_input_ppt.nc` | `ppt_noroeste_CORRUPTO_prueba_T2.nc` |
| `phase2_ppt_polygon_clip_1981_2025.nc` | `ppt_noroeste_shp_1981_2025.nc` |
| `phase2_pet_polygon_clip_1981_2025.nc` | `pet_noroeste_shp_1981_2025.nc` |
| `phase2_t2_corrupted_input_ppt.nc` | `ppt_noroeste_shp_CORRUPTO_prueba_T2.nc` |
| `study_area/northwest_mexico_states.shp` | `Noroeste/Noroeste.shp` |
