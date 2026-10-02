# Reference implementations

| Folder | Content |
|---|---|
| `phase1_python/` | Python reference for Phase 1 (rectangular clip), written by the authors with AI assistance. `ground_truth_hidroclima.py` and `ground_truth_hidroclima_2.py` are the corrected versions: the first version summed with `skipna=True` and turned all-NaN ocean cells into zeros, and the correction adds `min_count=1` to the temporal sums of T3 and T4. `ground_truth_hidroclima_2.py` only adds caching of the unified `pet` file. `generar_archivo_corrupto_T2.py` creates the corrupted Phase 1 T2 input (seed 42, same procedure as the reference). |
| `phase2_arcmap/` | ArcMap 10.8 / ArcPy (Python 2.7) procedure used as the Phase 2 reference (polygon clip, climatology, annual totals, water balance and trend statistics). |

Reference values of both phases are reported in Table 4 of the chapter.
