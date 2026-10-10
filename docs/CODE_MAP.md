# Results and the code that produces them

Results of the paper that this repository reproduces, with the workflow (see `docs/WORKFLOWS.md`) and the script that
writes the numbers. The other analyses of the paper are not part of this repository.

| result | runs | numbers written by |
|---|---|---|
| Main simulation: unpaired group, heterogeneous response and modality shift, data seeds 21-30 (errors, expression-change correlation, sign accuracy, spatial reproducibility, simple baselines, control reconstruction) | `scripts/run_condition.sh capture N main none`, `hetero N`, `capture N main orig` | `tables/sign_accuracy.py`, `tables/make_simulation_table.py` |
| Changed reference and observation conditions (reference not made of the ST cells, capture efficiency, depth, all combined) | `scripts/lists/base_runs.txt`, `scripts/lists/reuse_runs.txt` | `tables/make_condition_tables.py` |
| Predictions without transport and response-specific evaluation against the counterfactual change | the main simulation runs (rows `*_noT`, columns `resp_*` of `eval.csv`) | `pipeline/reference_rows.py`, `tables/make_condition_tables.py` |
| Accuracy of the estimated cell-type composition | `diag` stage of every run | `pipeline/composition_summary.py`, `tables/make_condition_tables.py` |
| Complex spatial structure, with and without capture efficiency | `scripts/generate_complex_data.sh`, `capture N complex orig`, `capture N complex none` | `tables/make_condition_tables.py` |
| Hyperparameter comparison of `lambda_sp_gamma` and `lambda_gamma` (data seeds 11-13) | `scripts/run_grid_queue.sh` | `tables/selection_grid_summary.py` |
| Mouse liver injury: zone-level comparison of predicted and observed changes, gene-category errors, clustering seeds, the other healthy section, later time points, depth | `realdata/liver/run.sh` | `realdata/liver/zone_eval.py`, `depth_check.py`, `summarize.py` |
| Mouse kidney injury: Havcr1 induction by region and cell type, anatomical zones, gene-category errors | `realdata/kidney/run.sh` | `realdata/kidney/region_eval.py`, `type_attribution.py`, `summarize.py` |

## Settings

- Loss weights: `pipeline/standard_settings.json`; `pipeline/core.py` (`canonical_source_audit`) checks that the code
  defaults agree with them before training. Other training settings (optimizer, epochs, batch size, optimal transport
  regularization, neighbour graph) are the defaults of `src/perturbed_spatial_omics/stmix.py` (`default_stmix_config`)
  and `transport.py` (`default_cellot_config`).
- Random seeds: the data seed sets every random stream of the simulation; `--seed` of `pipeline/run_condition.py` sets
  the PCA and Leiden clustering of the reference and the SIMO mapping; the initial weights and mini-batch order of all
  networks use the same random seed (1) in every run.
- Simulation design: `simulation/generation_config.json` and `simulation/complex_scenario/config.json`.
- Package versions: `environment/requirements.txt` and the Dockerfiles in `environment/`.
