"""Export perturbed-gene sign accuracy as CSV."""
import sys, pathlib
import anndata as ad, numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "pipeline"))
from reference_rows import dense, normlog

PSEUDO = 1e-6
COND = [  # scenario label, run directory, run name (file prefix), scenario data directory
    ("unpaired group", "results/conditions/capture/seed_{N}/main_cap_none", "main_cap_none", "results/conditions/capture/seed_{N}/main_cap_none/scenario"),
    ("heterogeneous response", "results/conditions/hetero/seed_{N}/hetero_main", "hetero_main", "results/simulation/data_seed_{N}/unpaired_group_heterogeneous"),
    ("modality shift", "results/conditions/capture/seed_{N}/main_cap_orig", "main_cap_orig", "results/simulation/data_seed_{N}/unpaired_group_modality"),
]


def lfc(ctrl_log, x_log):
    c = np.expm1(ctrl_log).clip(min=0).mean(axis=0); s = np.expm1(x_log).clip(min=0).mean(axis=0)
    return np.log2((s + PSEUDO) / (c + PSEUDO))


rows = []
for N in map(int, sys.argv[1:]):
    for cname, rdir, S, sdir in COND:
        R, D = pathlib.Path(rdir.format(N=N)), pathlib.Path(sdir.format(N=N)); P = R / "model/predictions"
        files = {"proposed": P / f"{S}_proposed_stim_raw_G8000_modelseed{N}.h5ad", "simo": P / f"{S}_simo_cellot_normlog_G8000_modelseed{N}__norepeat.h5ad"}
        if not all(f.exists() for f in files.values()):
            continue
        st = ad.read_h5ad(D / "st_ctrl_raw.h5ad"); genes = st.var_names.astype(str).to_numpy()
        y_ctrl = normlog(dense(st.X)); y_true = normlog(dense(ad.read_h5ad(D / "st_stim_raw.h5ad").X))
        mask = np.isin(genes, pd.read_csv(D / "perturbed_genes.csv")["gene_id"].astype(str))
        true_l = lfc(y_ctrl, y_true)[mask]
        for k, f in files.items():
            x = dense(ad.read_h5ad(f).X); x_log = normlog(x) if k == "proposed" else x
            pred_l = lfc(y_ctrl, x_log)[mask]
            rows.append({"condition": cname, "seed": N, "method": k, "sign_accuracy_log2fc_P": float(np.mean(np.sign(pred_l) == np.sign(true_l)))})
df = pd.DataFrame(rows)
out = pathlib.Path("results/tables/sign_accuracy.csv"); out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
