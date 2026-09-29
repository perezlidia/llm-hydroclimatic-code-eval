"""
T2 (Phase 2): runs the three model scripts WITHOUT modifying them and stores each model's output
under its own name, so that they do not overwrite each other.

Expects in the working directory:
    t2_gpt6astra.py  t2_claudeopus55.py  t2_gemini31pro.py
    ppt_noroeste_shp_CORRUPTO_prueba_T2.nc
Run:
    python export_t2_outputs.py
Produces:
    T2_claudeopus55.nc         cleaned output of Claude Opus 5.5 (includes its qc_flag variable)
    T2_gemini31pro.nc          'ppt_clean' of Gemini 3.1 Pro, taken from memory (its save line is commented out)
    T2_gpt6astra.nc            only if GPT-6 Astra decides to save (expected: NO, "LIMPIEZA INCOMPLETA")
    T2_report_gpt6astra.txt    GPT-6 Astra's own report
    T2_log_<model>.txt         everything each script printed
"""
import os
import runpy
import shutil
import subprocess
import sys
from contextlib import redirect_stdout
from io import StringIO

from config import BASE, T2_INPUT

os.chdir(BASE)                                    # Gemini opens its input with a relative path
SHARED_OUTPUT = BASE / "ppt_noroeste_shp_LIMPIO_T2.nc"   # name used by both GPT-6 Astra and Claude

if not T2_INPUT.exists():
    sys.exit(f"T2 input not found: {T2_INPUT}")


def move_previous_output():
    """If an output from an earlier run exists, rename it so it is not mistaken for a new one."""
    if SHARED_OUTPUT.exists():
        previous = SHARED_OUTPUT.with_name(SHARED_OUTPUT.stem + "_PREVIOUS.nc")
        shutil.move(SHARED_OUTPUT, previous)
        print(f"  (previous output moved to {previous.name})")


def run(script, model):
    """Runs the script unchanged in a separate process and stores what it prints."""
    print(f"\n===== {model.upper()}: python {script} =====")
    move_previous_output()
    r = subprocess.run([sys.executable, script], cwd=BASE, capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    output = r.stdout + ("\n--- stderr ---\n" + r.stderr if r.stderr.strip() else "")
    (BASE / f"T2_log_{model}.txt").write_text(output, encoding="utf-8")
    print(output.strip())
    print(f"  exit code: {r.returncode}")
    return r.returncode


# --- GPT-6 Astra: may refuse to save; that is its result
run("t2_gpt6astra.py", "gpt6astra")
if SHARED_OUTPUT.exists():
    shutil.move(SHARED_OUTPUT, BASE / "T2_gpt6astra.nc")
    print("  -> GPT-6 Astra DID save a file: T2_gpt6astra.nc")
else:
    print("  -> GPT-6 Astra did NOT save a cleaned file (model result, not an error)")
report = BASE / "reporte_limpieza_ppt_T2.txt"
if report.exists():
    shutil.copy(report, BASE / "T2_report_gpt6astra.txt")

# --- Claude Opus 5.5
run("t2_claudeopus55.py", "claudeopus55")
if SHARED_OUTPUT.exists():
    shutil.move(SHARED_OUTPUT, BASE / "T2_claudeopus55.nc")
    print("  -> saved as T2_claudeopus55.nc")
else:
    print("  -> Claude Opus 5.5 produced no file (see T2_log_claudeopus55.txt)")

# --- Gemini 3.1 Pro: the script does not save; it is run and its 'ppt_clean' is taken from memory
print("\n===== GEMINI31PRO: python t2_gemini31pro.py (in memory) =====")
buffer = StringIO()
try:
    with redirect_stdout(buffer):
        variables = runpy.run_path(str(BASE / "t2_gemini31pro.py"), run_name="__main__")
    print(buffer.getvalue().strip())
    variables["ppt_clean"].to_dataset(name="ppt").to_netcdf(BASE / "T2_gemini31pro.nc")
    print("  -> ppt_clean exported as T2_gemini31pro.nc (Gemini left its save line commented out)")
except Exception as e:
    print(buffer.getvalue().strip())
    print(f"  -> ERROR while running Gemini 3.1 Pro: {e}")
(BASE / "T2_log_gemini31pro.txt").write_text(buffer.getvalue(), encoding="utf-8")

print("\nDone. Next: python export_t2_gpt_in_memory.py")
