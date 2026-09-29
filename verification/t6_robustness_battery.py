"""
T6 (Phase 2): extended robustness test.
Submits the loading functions of the three models, WITHOUT MODIFYING THEM, to the same battery of
ten cases: the two cases of the prompt, positive controls (valid files that must be accepted) and
additional defective inputs.

Expects in the working directory:
    t6_gpt6astra.py  t6_claudeopus55.py  t6_gemini31pro.py
    (optional) ppt_noroeste_shp_1981_2025.nc   -> real study file used as a positive control
Run:
    python t6_robustness_battery.py
Produces:
    T6_robustness_matrix.csv   and   figures/fig8_t6_robustness.png / .pdf
"""
import csv
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
FIG = BASE / "figures"
FIG.mkdir(exist_ok=True)

import t6_gpt6astra      # noqa: E402  (importing does not run their tests: they are under __main__)
import t6_claudeopus55   # noqa: E402
import t6_gemini31pro    # noqa: E402

# Function of each model and the exception types its own code declares
MODELS = {
    "GPT-6 Astra":     (t6_gpt6astra.cargar_netcdf_seguro, (t6_gpt6astra.ErrorNetCDF,)),
    "Claude Opus 5.5": (t6_claudeopus55.cargar_netcdf_seguro, (t6_claudeopus55.ErrorCargaNetCDF,)),
    "Gemini 3.1 Pro":  (t6_gemini31pro.cargar_netcdf_climatico, (FileNotFoundError, ValueError)),
}

# Typical fragments of internal library errors (NetCDF/HDF5/xarray/OS). If present, the message
# exposes the internal detail instead of abstracting it. Every flagged message was reviewed manually.
LEAK = ["[Errno", "Errno ", "NetCDF: ", "IO backends", "Unable to", "h5py", "OSError",
        "Traceback", "engine=", "KeyError", "Consider explicitly", "HDF error"]


def test_dataset(variable="ppt", with_time=True):
    lat, lon = [25.0, 25.5], [-110.0, -109.5]
    rng = np.random.default_rng(0)
    if with_time:
        t = pd.date_range("2000-01-01", periods=3, freq="MS")
        return xr.Dataset({variable: (("time", "lat", "lon"), rng.random((3, 2, 2), dtype="float32"))},
                          coords={"time": t, "lat": lat, "lon": lon})
    return xr.Dataset({variable: (("lat", "lon"), rng.random((2, 2), dtype="float32"))},
                      coords={"lat": lat, "lon": lon})


def build_cases(tmp):
    cases = []   # (id, description, path, must_be_accepted)
    cases.append(("a", "Non-existent file [prompt]", tmp / "does_not_exist.nc", False))
    p = tmp / "text.nc"
    p.write_text("lat,lon,time,ppt\n25.8,-108.9,1981-01-01,3.2\n", encoding="utf-8")
    cases.append(("b", "Plain text renamed to .nc [prompt]", p, False))
    p4 = tmp / "valid_nc4.nc"
    test_dataset().to_netcdf(p4, format="NETCDF4")
    cases.append(("c", "Valid NetCDF-4 [positive control]", p4, True))
    p = tmp / "valid_nc3.nc"
    test_dataset().to_netcdf(p, format="NETCDF3_CLASSIC")
    cases.append(("d", "Valid classic NetCDF-3 [positive control]", p, True))
    real = BASE / "ppt_noroeste_shp_1981_2025.nc"
    if real.exists():
        cases.append(("e", "Actual study file [positive control]", real, True))
    p = tmp / "empty.nc"
    p.write_bytes(b"")
    cases.append(("f", "Empty file (0 bytes)", p, False))
    p = tmp / "folder.nc"
    p.mkdir()
    cases.append(("g", "Folder with .nc extension", p, False))
    p = tmp / "no_variable.nc"
    test_dataset(variable="tmax").to_netcdf(p)
    cases.append(("h", "Valid NetCDF without 'ppt' or 'pet'", p, False))
    p = tmp / "no_time.nc"
    test_dataset(with_time=False).to_netcdf(p)
    cases.append(("i", "Valid NetCDF without 'time' dimension", p, False))
    p = tmp / "truncated.nc"
    data = p4.read_bytes()
    p.write_bytes(data[: len(data) // 2])
    cases.append(("j", "NetCDF-4 truncated by half", p, False))
    return cases


def evaluate(function, types, path, must_accept):
    try:
        ds = function(path)
    except types as e:
        msg = str(e)
        if must_accept:
            return "rejects_valid", type(e).__name__, msg
        return ("rejects_leak" if any(k in msg for k in LEAK) else "rejects_ok"), type(e).__name__, msg
    except Exception as e:          # exception not handled by the function
        return "uncontrolled", type(e).__name__, str(e)
    try:
        ds.close()
    except Exception:
        pass
    return ("accepts_ok" if must_accept else "accepts_invalid"), "-", "(file loaded)"


CATEGORY = {  # colour code, cell label
    "accepts_ok":      (0, "Accepts ✓"),
    "rejects_ok":      (0, "Rejects ✓"),
    "rejects_leak":    (1, "Rejects ✓\nexposes internal detail"),
    "accepts_invalid": (2, "Accepts ✗"),
    "rejects_valid":   (2, "Rejects ✗"),
    "uncontrolled":    (2, "Uncontrolled error ✗"),
}

rows = []
with tempfile.TemporaryDirectory(prefix="t6_", ignore_cleanup_errors=True) as folder:
    cases = build_cases(Path(folder))
    for cid, desc, path, must in cases:
        print(f"\n({cid}) {desc}  -> expected: {'ACCEPT' if must else 'REJECT'}")
        for name, (function, types) in MODELS.items():
            result, etype, msg = evaluate(function, types, path, must)
            print(f"  {name:16s} {result:16s} [{etype}] {msg}")
            rows.append(dict(case=cid, description=desc, must_accept=must, model=name,
                             result=result, exception=etype, message=msg))

with open(BASE / "T6_robustness_matrix.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

print("\nSUMMARY (correct / cases)")
for name in MODELS:
    r = [x["result"] for x in rows if x["model"] == name]
    ok = sum(x in ("accepts_ok", "rejects_ok", "rejects_leak") for x in r)
    leak = sum(x == "rejects_leak" for x in r)
    print(f"  {name:16s} {ok}/{len(r)} correct; {leak} exposing internal library detail")

# ---- Fig. 8: cases x models matrix
ids = [c[0] for c in cases]
descs = [f"({c[0]}) {c[1]}" for c in cases]
names = list(MODELS)
codes = np.zeros((len(ids), len(names)))
texts = [["" for _ in names] for _ in ids]
for row in rows:
    i, j = ids.index(row["case"]), names.index(row["model"])
    codes[i, j], texts[i][j] = CATEGORY[row["result"]]

colours = ["#a6d96a", "#fee08b", "#f46d43"]
fig, ax = plt.subplots(figsize=(8.2, 0.55 * len(ids) + 1.6), constrained_layout=True)
ax.imshow(codes, cmap=ListedColormap(colours), vmin=-0.5, vmax=2.5, aspect="auto")
for i in range(len(ids)):
    for j in range(len(names)):
        ax.text(j, i, texts[i][j], ha="center", va="center", fontsize=8.5)
ax.set_xticks(range(len(names)), names, fontsize=10)
ax.xaxis.tick_top()
ax.set_yticks(range(len(ids)), descs, fontsize=9)
ax.set_xticks(np.arange(-0.5, len(names)), minor=True)
ax.set_yticks(np.arange(-0.5, len(ids)), minor=True)
ax.grid(which="minor", color="white", linewidth=2)
ax.tick_params(which="both", length=0)
fig.legend(handles=[Patch(color=colours[0], label="Correct behaviour"),
                    Patch(color=colours[1], label="Rejects, but exposes the internal library error"),
                    Patch(color=colours[2], label="Incorrect behaviour")],
           loc="lower center", ncol=1, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.12))
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"fig8_t6_robustness.{ext}", dpi=300, bbox_inches="tight")
print(f"\nFigure saved to {FIG / 'fig8_t6_robustness.png'}")
