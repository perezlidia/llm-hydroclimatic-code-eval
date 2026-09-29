"""
Builds the flat working directory in which the Phase 2 model scripts and the verification
scripts are run (all of them expect their inputs in the same folder).

Usage:
    python prepare_workdir.py --data <unzipped data folder> --out <workdir>

The data folder is the unzipped data package (see data/README.md). Data files are published
with English names, but the model scripts read their inputs with the original Spanish names
given in the prompts; this script therefore copies those inputs under the names the models
expect (INPUT_NAMES below). The model scripts are copied byte for byte; they are never modified.
"""
import argparse
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True, type=Path)
parser.add_argument("--out", required=True, type=Path)
args = parser.parse_args()
out = args.out
out.mkdir(parents=True, exist_ok=True)

# 1) Phase 2 model scripts T2-T6 (T1 scripts use absolute Windows paths; their outputs are provided as data)
for task in range(2, 7):
    for script in sorted((REPO / "generated_code" / "phase2" / f"t{task}").glob("*.py")):
        shutil.copy2(script, out / script.name)
        print(f"model script   {script.relative_to(REPO)}")

# 2) Verification and figure scripts
for script in sorted((REPO / "verification").glob("*.py")):
    if script.name != "prepare_workdir.py":
        shutil.copy2(script, out / script.name)
        print(f"verification   {script.name}")

# 3) Input data (files and sub-folders). Model inputs are restored to the file names used in
#    the prompts, because the unmodified model scripts look for those names.
INPUT_NAMES = {
    "phase2_ppt_polygon_clip_1981_2025.nc": "ppt_noroeste_shp_1981_2025.nc",
    "phase2_pet_polygon_clip_1981_2025.nc": "pet_noroeste_shp_1981_2025.nc",
    "phase2_t2_corrupted_input_ppt.nc": "ppt_noroeste_shp_CORRUPTO_prueba_T2.nc",
    "phase1_ppt_rectangular_clip_1981_2025.nc": "ppt_noroeste_1981_2025.nc",
}
for item in sorted(args.data.iterdir()):
    target = out / INPUT_NAMES.get(item.name, item.name)
    if item.is_dir():
        shutil.copytree(item, target, dirs_exist_ok=True)
    else:
        shutil.copy2(item, target)
    print(f"data           {item.name}  ->  {target.name}")

print(f"\nWorking directory ready: {out}\nSee README.md, section 'Reproducing the results'.")
