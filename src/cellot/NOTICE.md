# Third-party code: CellOT

This directory contains part of CellOT, taken from https://github.com/bunnech/cellot.

CellOT is distributed under the BSD 3-Clause License. Its license text, including the copyright notice, is reproduced
unchanged in `LICENSE` in this directory.

## Files taken

Only the modules imported by this repository are included (installed as the top-level package `cellot` by
`pip install .`): `__init__.py`, `data/` (`cell.py`, `utils.py`), `losses/`
(`mmd.py`), `models/` (`ae.py`, `cellot.py`), `networks/` (`icnns.py`) and `utils/helpers.py`, each with its package
`__init__.py`. CellOT's training, transport, evaluation, preprocessing, flag and visualization modules are not included.

## Local changes

Compared with the upstream `main` branch (commit `522d2b9`), ten of the thirteen files are identical and three differ:

- `data/utils.py`: `cast_dataset_to_loader` returns an empty loader structure when the dataset is empty; trailing
  commented-out lines were added.
- `models/cellot.py`: comments were added, together with a commented-out variant of `compute_loss_g` and a helper
  `compute_spatial_consistency_loss` that this repository does not use. The functions used by this repository
  (`load_networks`, `load_cellot_model`, `compute_loss_f`, `compute_loss_g`, `compute_w2_distance`) differ from
  upstream only in comments.
- `utils/helpers.py`: `MutableMapping` is imported from `collections.abc` (required for Python 3.10).
