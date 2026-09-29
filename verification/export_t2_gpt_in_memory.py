"""
T2 (Phase 2): extracts the IN-MEMORY result computed by GPT-6 Astra (which the model refused to
save), by calling its own limpiar() function without modifying its code. Used only for plotting.

Run:
    python export_t2_gpt_in_memory.py
Produces:
    T2_gpt6astra_in_memory.nc
"""
import sys
import xarray as xr
from config import BASE, T2_INPUT

sys.path.insert(0, str(BASE))
import t2_gpt6astra  # only defines functions; its main() does not run on import

# Opened exactly as GPT-6 Astra's own script opens it
with xr.open_dataset(T2_INPUT, decode_times=True, mask_and_scale=True) as source:
    ds = source.load()

cleaned, report = t2_gpt6astra.limpiar(ds["ppt"])
for key, value in report.items():
    print(f"{key}: {value}")

cleaned = cleaned.copy()
cleaned.attrs = {"note": "In-memory result of t2_gpt6astra.limpiar(); the model did NOT save it "
                         "because it reported LIMPIEZA INCOMPLETA (incomplete cleaning)"}
cleaned.to_dataset(name="ppt").to_netcdf(BASE / "T2_gpt6astra_in_memory.nc")
print("\nSaved: T2_gpt6astra_in_memory.nc (plotting only; GPT-6 Astra produced no file)")
