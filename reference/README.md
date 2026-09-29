# Reference implementations

| Folder | Content |
|---|---|
| `phase1_python/` | Python reference for Phase 1 (rectangular clip). The first version summed with `skipna=True` and turned all-NaN ocean cells into zeros; the corrected version adds `min_count=1` to the temporal sums of T3 and T4. Both versions are kept. |
| `phase2_arcmap/` | ArcMap 10.8 / ArcPy (Python 2.7) procedure used as the Phase 2 reference (polygon clip, climatology, annual totals, water balance and trend statistics). |

Reference values of both phases are reported in Table 4 of the chapter.
