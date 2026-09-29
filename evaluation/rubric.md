# Evaluation rubric

Each model delivery was scored per task with the seven criteria below (maximum 24 points per task,
144 per model and phase). Scores per task are listed in `scores.csv`.

| Criterion | Description | Points |
|---|---|---|
| Numerical correctness | Agreement with the reference values for the requested quantity | 0–5 |
| Executability | Initial execution of the code exactly as received, without modifications | 0–3 |
| Edge cases | Handling of NaN, corrupted data and coordinate misalignment, depending on the task | 0–5 |
| Error handling | Failure detection and clear, descriptive messages | 0–3 |
| Good practices | Clarity, organization, comments, style and use of vectorized operations | 0–3 |
| Prompt fidelity | Fulfilment of the explicit requirements of the task | 0–3 |
| Efficiency | Execution time relative to the data volume (assessed qualitatively) | 0–2 |
| **Maximum per task** | | **24** |

Notes

- Differences caused by methodological choices (e.g. area weighting by the cosine of latitude)
  were distinguished from calculation errors.
- Phase 2, T1, Gemini 3.1 Pro: scored after removing the unsupported argument `clip_box=False`;
  the Phase 2 total of this model therefore includes a repaired delivery.
- Phase 2, T2: after the cell-by-cell verification identified 394 study-area cells set to NaN over
  the whole series, the same one-point deduction in *Edge cases* was applied to the three models,
  none of which detected them.
- No systematic runtime or memory measurements were recorded; efficiency was assessed qualitatively.
