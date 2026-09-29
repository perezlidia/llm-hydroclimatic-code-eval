"""
T4 (Phase 2): exports each model's multi-year water-balance map PER CELL without modifying the
model code. None of the delivered scripts saved this map.

Expects in the working directory:
    t4_gpt6astra.py  t4_claudeopus55.py  t4_gemini31pro.py
    ppt_noroeste_shp_1981_2025.nc  pet_noroeste_shp_1981_2025.nc
Run:
    python export_t4_maps.py
Produces:
    T4_gpt6astra.nc  T4_claudeopus55.nc  T4_gemini31pro.nc
"""
import os
import runpy
import sys
import xarray as xr

from config import BASE, PPT, PET
from spatial_figures import export_result

os.chdir(BASE)                     # the model scripts read their inputs with relative paths
sys.path.insert(0, str(BASE))

# --- Claude Opus 5.5: main() already returns the per-cell map (bh_multianual_mapa)
import t4_claudeopus55
_, _, map_claude, _ = t4_claudeopus55.main()
export_result(map_claude, "T4", "claudeopus55")

# --- GPT-6 Astra: calcular() returns 'anual'; the map is obtained with the SAME expression that
#     GPT-6 Astra uses internally:  anual.mean("year", skipna=False).where(area)
import t4_gpt6astra
with xr.open_dataset(PPT) as ds_ppt, xr.open_dataset(PET) as ds_pet:
    _, annual, _, _, _ = t4_gpt6astra.calcular(ds_ppt, ds_pet)
    area = annual.notnull().any("year")
    map_gpt = annual.mean("year", skipna=False).where(area).load()
export_result(map_gpt, "T4", "gpt6astra")

# --- Gemini 3.1 Pro: module-level script; it is run in full and its 'bal_anual' is taken.
#     Gemini never computes a per-cell multi-year map, so it is derived here.
gemini_vars = runpy.run_path(str(BASE / "t4_gemini31pro.py"))
map_gemini = gemini_vars["bal_anual"].mean("year")
export_result(map_gemini, "T4", "gemini31pro")

print("Done. Next: python spatial_figures.py T4")
