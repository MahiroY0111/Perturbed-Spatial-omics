"""Perturbation responses in spatial transcriptomics by optimal transport map transfer.

  stmix        spot model: each spot as a mixture of cell-type pseudo-cells in the latent space of the reference
  autoencoder  autoencoder of the reference scRNA-seq (CellOT's AutoEncoder)
  transport    CellOT map between control and stimulated latents, transfer to the pseudo-cells and decoding
"""

__version__ = "1.0.0"
