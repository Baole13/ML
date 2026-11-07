# Benchmark results

`preliminary_quick_mode/` contains a transcription of the outputs saved in the
notebook. It is a 20-class Quick Mode smoke test (2 ResNet epochs), not the
official benchmark. Keep it labelled preliminary in slides and the report.
The two plots under `preliminary_quick_mode/figures/` are also Quick Mode
outputs and should be labelled accordingly.

The official full run belongs in `official_full/` and should contain at least:

- `benchmark_results.csv`
- `integrity_summary.csv`
- `fixed_clean_split.csv`
- `resnet_test_predictions.csv`
- `run_config.json` with seed, mode, class/sample counts, device and key package versions
- `figures/` for model comparison, confusion matrix, and retrieval/error examples

Create these outputs by restarting the notebook runtime, setting
`QUICK_MODE = False`, and running all cells from the beginning. Do not combine
metrics from different runs or splits. The official serving package must be
exported from the same full-run DINOv2 gallery and classifier.
