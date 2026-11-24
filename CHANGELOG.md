# Changelog

All code modifications and fixes made during execution are tracked here.

Format: Date | File | Line(s) | Reason | Description
---
- 2026-11-18 | `src/prepare_data.py` | 72-108 | Dataset layout compatibility | Added `resolve_paths()` function to handle both MultiClinAI v1.1 (`MultiClinNER/<lang>/.../txt` & `ann`) and v1.0 structures, selecting the 750 SPACCC documents for benchmark alignment.
- 2026-11-18 | `src/evaluate.py` | 10, 44 | Keras 3 deserialization security policy & module loading | Added `import models` so custom registered layers are available during model loading, and added `safe_mode=False` to `load_model(run / "model.keras", safe_mode=False)`.
- 2026-11-18 | `src/models.py` | 5-27 | Keras 3 Lambda serialization & name scoping | Replaced anonymous untyped lambdas in `_mask` and `_attn_mask` with named functions decorated with `@keras.saving.register_keras_serializable(package="MultiClinAI")` and explicit output shapes (`_pad_mask_shape`, `_attn_mask_shape`). This completely resolves both `NotImplementedError` (inferred shape) and `NameError: name 'tf' is not defined` when deserializing saved `.keras` models.
- 2026-11-24 | `src/run_experiments.py` | Full file | Automated pipeline execution | Created automated multi-model multi-seed execution harness that runs training and evaluation sequentially, logs all console outputs, skips already-evaluated runs, and generates final reports.
- 2026-11-24 | `src/export_results.py` | Full file | Downloadable reports request | Added script to build styled multi-tab Excel spreadsheet (`MultiClinAI_Clinical_NER_Results.xlsx`) and publication-grade vector graphics PDF report (`MultiClinAI_Clinical_NER_Report.pdf`) with performance bar charts and error bars.
