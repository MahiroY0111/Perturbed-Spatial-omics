# R environment for the Splatter cell pool of the simulation (simulation/create_pool.R):
# Bioconductor 3.20 (R 4.4) with Splatter 1.30.0.
FROM bioconductor/bioconductor_docker:RELEASE_3_20

RUN R -q -e 'BiocManager::install(c("splatter", "SingleCellExperiment", "SummarizedExperiment", "HDF5Array", "scater", "jsonlite"), ask = FALSE, update = FALSE)' \
 && R -q -e 'stopifnot(as.character(packageVersion("splatter")) == "1.30.0"); library(HDF5Array); library(jsonlite)'

ENV HOME=/tmp
WORKDIR /workspace
ENTRYPOINT []
CMD ["bash"]
