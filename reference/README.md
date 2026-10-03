# Reference implementations

| Folder | Content |
|---|---|
| `phase1_python/` | Python reference for Phase 1 (rectangular clip), written by the authors with AI assistance (comments and messages translated into English; input file names kept as used in the study). `reference_phase1.py` and `reference_phase1_v2.py` are the corrected versions: the first version summed with `skipna=True` and turned all-NaN ocean cells into zeros, and the correction adds `min_count=1` to the temporal sums of T3 and T4. `reference_phase1_v2.py` only adds caching of the unified `pet` file. `make_corrupted_t2_input_phase1.py` creates the corrupted Phase 1 T2 input (seed 42, same procedure as the reference). |

The Phase 2 reference was computed manually in ArcMap 10.8 with geoprocessing tools; its procedure
is not distributed. The resulting reference rasters (`arcmap/` in the data package) are provided
for comparison. Reference values of both phases are reported in Table 4 of the chapter.
