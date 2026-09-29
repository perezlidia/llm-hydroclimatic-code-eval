"""
Shared configuration for the verification and figure scripts.

All scripts are designed to run inside a single flat working directory
("workdir") that contains the model scripts, the input data and these
verification scripts. Create it with `prepare_workdir.py`.
"""
from pathlib import Path

BASE = Path(__file__).resolve().parent          # the working directory
ARCMAP = BASE / "arcmap"                        # Phase 2 ArcMap reference rasters
FIG = BASE / "figures"                          # output figures
SHP = BASE / "study_area" / "northwest_mexico_states.shp"   # state polygons (optional, outlines only)
T1_ARCMAP = ARCMAP / "t1_ppt_1981_clip_arcmap.tif"          # ArcMap clip (1981, 12 bands)
T4_ARCMAP = ARCMAP / "t4_water_balance_arcmap.tif"          # ArcMap mean multi-year water balance

# Common Phase 2 inputs, under the names the model scripts expect (restored by prepare_workdir.py)
PPT = BASE / "ppt_noroeste_shp_1981_2025.nc"
PET = BASE / "pet_noroeste_shp_1981_2025.nc"
T2_INPUT = BASE / "ppt_noroeste_shp_CORRUPTO_prueba_T2.nc"   # corrupted input given to the models in T2

# File-name key -> label used in figures and tables
MODELS = {
    "gpt6astra": "GPT-6 Astra",
    "claudeopus55": "Claude Opus 5.5",
    "gemini31pro": "Gemini 3.1 Pro",
}

DPI = 300
