# Workflows

All commands run from the repository root. Python steps run in the containers of `environment/compose.yaml`; the job
scripts call them for you. Training and prediction use the CPU image `py` by default; set `PYSVC=pygpu` to use an
NVIDIA GPU (same package versions with PyTorch 2.7.1; results differ slightly from the CPU build). Every job script
skips steps whose output exists, so a stopped job resumes with the same command. Threads per container: `TH`
(default 4) or the last argument of the job scripts.

Shorthand used below: `dpy` = `docker compose -f environment/compose.yaml run --rm py`.

## 1. Environment

```bash
docker compose -f environment/compose.yaml --profile simo --profile r build py simo r
docker compose -f environment/compose.yaml --profile gpu build pygpu          # optional
dpy python environment/smoke_test.py                                            # package versions and imports
```

## 2. Simulated data

```bash
scripts/generate_data.sh evaluation 21 22 23 24 25 26 27 28 29 30    # final evaluation, results/simulation/data_seed_N/
scripts/generate_complex_data.sh 21 22 23 24 25 26 27 28 29 30       # complex spatial structure, results/complex_scenario/seed_N/
```

Each data seed holds a Splatter pool of 50,000 cells and 8,000 genes (`common/`) and the scenarios built from it, among
them `unpaired_group`, `unpaired_group_heterogeneous` and `unpaired_group_modality`. The generation settings are in
`simulation/generation_config.json`; the seeds of all random streams follow from the data seed.

## 3. Simulated conditions

One condition and one data seed per call of `scripts/run_condition.sh` (usage at the top of the script). The reported
runs are listed in `scripts/lists/`; run the base conditions first, because the others reuse their autoencoder and
CellOT (same single-cell reference):

```bash
scripts/run_queue.sh scripts/lists/base_runs.txt 3          # 3 jobs at a time
scripts/run_queue.sh scripts/lists/reuse_runs.txt 3
```

| condition (`run_condition.sh` arguments) | scenario |
|---|---|
| `capture N main none` | unpaired group (no capture-efficiency model) |
| `hetero N` | heterogeneous response |
| `capture N main orig` | modality shift (gene-wise capture efficiency Beta(8, 2)) |
| `capture N main beta:2:2`, `beta:1:1` | stronger gene-to-gene differences of capture efficiency |
| `capture N main const:P` | the same capture probability P for all genes (lower depth) |
| `refctrl N F`, `refshift N F`, `refsub N F` | replaced single-cell references; F scales the responding group |
| `realistic N 1.0 betaconst:8:2:0.03` | replaced reference, gene-wise capture efficiency and low depth together |
| `capture N complex orig`, `capture N complex none` | complex spatial structure, with and without capture efficiency |

Each run writes `results/conditions/<family>/seed_N/<name>/` with the condition data, the SIMO mappings, the models,
the predictions, logs and `eval.csv`. `eval.csv` has one row per method and the metrics described at the top of
`pipeline/reference_rows.py`: errors against the observed stimulated ST (`RMSE_all`, `RMSE_P` for the perturbed genes,
`RMSE_Pc` for the others), expression-change correlations (`ms_*`), and the response-specific metrics (`resp_*`),
which compare each method's predicted change with the change obtained by applying the generating perturbation to the
cells of each control spot. Rows `*_noT` are the same methods with the transport replaced by the identity map.

## 4. Tables of the simulations

```bash
dpy python pipeline/summarize_runs.py --root results/conditions           # results/conditions/summary/all_rows.csv
dpy python pipeline/composition_summary.py --root results/conditions      # results/conditions/summary/composition.csv
dpy python tables/sign_accuracy.py 21 22 23 24 25 26 27 28 29 30          # results/tables/sign_accuracy.csv
dpy python tables/make_simulation_table.py                                # results/tables/simulation_table.tex
dpy python tables/make_condition_tables.py                                # results/tables/condition_tables.tex
```

`make_simulation_table.py --strict` stops when a run of the main table is missing.

## 5. Hyperparameter comparison

Twelve settings of the spatial penalty weight `lambda_sp_gamma` (0.01, 0.1, 1, 5) and of the weight of the spot/type
offsets `lambda_gamma` (0.01, 0.1, 1) on data seeds 11-13 and the three scenarios; all other settings as in
`pipeline/standard_settings.json`.

```bash
scripts/run_grid_queue.sh 3                     # data of seeds 11-13, then the 9 chains of 12 settings
dpy python tables/selection_grid_summary.py     # results/selection_grid/summary/grid_summary.md
```

## 6. Public data

```bash
realdata/liver/fetch.sh && realdata/liver/run.sh          # results/liver_apap/ (figures/ holds the summary)
realdata/kidney/fetch.sh && realdata/kidney/run.sh        # results/kidney_iri/
```

The scripts download from GEO, build the inputs, run the method with the standard settings and with
`lambda_mix_gene = 1.0`, run SIMO+CellOT, compare predicted and observed changes by liver zone or kidney region, and
write the summaries (`summarize.py`). The comments at the top of `realdata/*/run.sh` list the runs.
