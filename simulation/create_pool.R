#!/usr/bin/env Rscript

# M2 simulation v2: Splatter master pool generation only.
# The formal pipeline starts here; no legacy RDS/H5AD is read.

suppressPackageStartupMessages({
  library(splatter)
  library(SingleCellExperiment)
  library(SummarizedExperiment)
  library(HDF5Array)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop("Usage: Rscript simulation/create_pool.R CONFIG_JSON OUTPUT_DIR")
}

config_path <- normalizePath(args[[1]], mustWork = TRUE)
output_dir <- normalizePath(args[[2]], mustWork = FALSE)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
config <- jsonlite::fromJSON(config_path, simplifyVector = TRUE)

d <- config$dimensions
s <- config$seeds
p <- config$splatter

stopifnot(
  d$n_genes == 8000L,
  d$splatter_pool_cells == 50000L,
  d$n_groups == 9L,
  length(p$group_prob) == d$n_groups,
  abs(sum(p$group_prob) - 1) < 1e-12
)

RNGkind(kind = "Mersenne-Twister", normal.kind = "Inversion", sample.kind = "Rejection")
set.seed(s$seed_splatter)

# Every Splat parameter used by this groups simulation is explicit. Path
# parameters are recorded too, although they are inactive for method="groups".
params <- newSplatParams(
  nGenes = as.integer(d$n_genes),
  batchCells = as.integer(d$splatter_pool_cells),
  batch.facLoc = p$batch_fac_loc,
  batch.facScale = p$batch_fac_scale,
  batch.rmEffect = p$batch_rm_effect,
  mean.shape = p$mean_shape,
  mean.rate = p$mean_rate,
  lib.loc = p$lib_loc,
  lib.scale = p$lib_scale,
  lib.norm = p$lib_norm,
  out.prob = p$out_prob,
  out.facLoc = p$out_fac_loc,
  out.facScale = p$out_fac_scale,
  group.prob = as.numeric(p$group_prob),
  de.prob = rep(p$de_prob, d$n_groups),
  de.downProb = rep(p$de_down_prob, d$n_groups),
  de.facLoc = rep(p$de_fac_loc, d$n_groups),
  de.facScale = rep(p$de_fac_scale, d$n_groups),
  bcv.common = p$bcv_common,
  bcv.df = p$bcv_df,
  dropout.type = p$dropout_type,
  dropout.mid = p$dropout_mid,
  dropout.shape = p$dropout_shape,
  path.from = rep(p$path_from, d$n_groups),
  path.nSteps = rep(as.integer(p$path_n_steps), d$n_groups),
  path.skew = rep(p$path_skew, d$n_groups),
  path.nonlinearProb = p$path_nonlinear_prob,
  path.sigmaFac = p$path_sigma_fac,
  seed = as.integer(s$seed_splatter)
)

message("Generating a new Splatter pool; no legacy simulation file is read.")
sce <- splatSimulate(
  params = params,
  method = "groups",
  sparsify = isTRUE(p$sparsify),
  verbose = TRUE
)

counts <- SummarizedExperiment::assay(sce, "counts")
stopifnot(
  nrow(counts) == d$n_genes,
  ncol(counts) == d$splatter_pool_cells,
  is.integer(counts),
  min(counts) >= 0
)

gene_ids <- sprintf("gene_%04d", seq_len(nrow(counts)))
cell_ids <- sprintf("pool_cell_%05d", seq_len(ncol(counts)))
rownames(counts) <- gene_ids
colnames(counts) <- cell_ids
true_group <- as.character(colData(sce)$Group)
stopifnot(length(unique(true_group)) == d$n_groups)

pool_h5 <- file.path(output_dir, "splatter_pool_counts.h5")
if (file.exists(pool_h5)) {
  stop("Refusing to overwrite existing Splatter pool: ", pool_h5)
}

# HDF5Array writes R's genes x cells matrix. h5py sees the same dataset as
# cells x genes, which is the AnnData orientation used by the Python stage.
HDF5Array::writeHDF5Array(
  counts,
  filepath = pool_h5,
  name = "X",
  chunkdim = c(512L, 256L),
  level = 6L,
  as.sparse = FALSE,
  with.dimnames = FALSE,
  verbose = TRUE
)

write.csv(
  data.frame(
    pool_cell_id = cell_ids,
    true_group = true_group,
    stringsAsFactors = FALSE
  ),
  file.path(output_dir, "splatter_pool_cell_metadata.csv"),
  row.names = FALSE,
  quote = TRUE
)
write.csv(
  data.frame(gene_id = gene_ids, stringsAsFactors = FALSE),
  file.path(output_dir, "gene_metadata.csv"),
  row.names = FALSE,
  quote = TRUE
)

capture.output(params, file = file.path(output_dir, "splatter_effective_params.txt"))
capture.output(sessionInfo(), file = file.path(output_dir, "sessionInfo.txt"))

pool_summary <- list(
  generated_from_splatter = TRUE,
  legacy_input_used = FALSE,
  n_genes = nrow(counts),
  n_cells = ncol(counts),
  count_class = class(counts),
  count_is_integer = is.integer(counts),
  count_min = min(counts),
  count_max = max(counts),
  group_counts = as.list(table(true_group)),
  seed_splatter = s$seed_splatter,
  splatter_version = as.character(packageVersion("splatter")),
  scater_version = if (requireNamespace("scater", quietly = TRUE)) as.character(packageVersion("scater")) else NA_character_,
  SingleCellExperiment_version = as.character(packageVersion("SingleCellExperiment")),
  Matrix_version = as.character(packageVersion("Matrix")),
  R_version = as.character(getRversion())
)
jsonlite::write_json(
  pool_summary,
  file.path(output_dir, "splatter_pool_audit.json"),
  pretty = TRUE,
  auto_unbox = TRUE
)

message("Splatter pool complete: ", pool_h5)
