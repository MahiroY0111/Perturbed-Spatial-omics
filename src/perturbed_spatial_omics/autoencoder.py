import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt

from cellot.models.ae import AutoEncoder
from cellot.data.cell import AnnDataDataset


def sparse_collate_fn(batch):
    """スパース行列 or dense array をテンソルに変換するヘルパー関数。"""
    tensors = []
    for x in batch:
        if hasattr(x, "toarray"):
            dense_array = x.toarray()
        else:
            dense_array = x
        dense_array = np.asarray(dense_array).squeeze()
        tensor = torch.FloatTensor(dense_array)
        tensors.append(tensor)
    return torch.stack(tensors)


def train_cellot_autoencoder(
    adata_hvg,
    latent_dim: int = 50,
    hidden_units: list[int] | tuple[int, ...] = (128, 128),
    n_epochs: int = 300,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    device: torch.device | None = None,
    obsm_key: str = "X_cellot_ae",
    plot_loss: bool = True,
):
    """cellot の AutoEncoder を用いて VAE 風表現を学習し、潜在表現を .obsm に格納する高レベル関数。

    Returns
    -------
    model : AutoEncoder
        学習済みモデル
    loss_history : list[float]
        各エポックの平均損失
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = adata_hvg.n_vars

    model = AutoEncoder(
        input_dim,
        hidden_units=list(hidden_units),
        latent_dim=latent_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Dataset / DataLoader 準備
    dataset = AnnDataDataset(adata_hvg)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=sparse_collate_fn,
    )

    loss_history: list[float] = []

    print(f"\n--- Starting AutoEncoder training for {n_epochs} epochs on device {device} ---")
    model.train()

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for batch_data in tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}"):
            batch_data = batch_data.to(device)

            optimizer.zero_grad(set_to_none=True)

            loss, _, _ = model(batch_data)
            loss = loss.mean()

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{n_epochs}, Average Loss: {avg_loss:.4f}")

    print("--- AutoEncoder training finished ---")

    if plot_loss:
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, n_epochs + 1), loss_history, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Average Loss")
        plt.title("AutoEncoder Training Loss over Epochs")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # 潜在表現の計算
    print("\n--- Encoding data to latent space ---")
    model.eval()

    if hasattr(adata_hvg.X, "toarray"):
        full_data_tensor = torch.tensor(adata_hvg.X.toarray(), dtype=torch.float32).to(device)
    else:
        full_data_tensor = torch.tensor(adata_hvg.X, dtype=torch.float32).to(device)

    with torch.no_grad():
        latent_vectors = model.encode(full_data_tensor)

    latent_vectors_np = latent_vectors.cpu().numpy()
    adata_hvg.obsm[obsm_key] = latent_vectors_np

    print(f"Latent vectors stored in adata_hvg.obsm['{obsm_key}'] with shape {latent_vectors_np.shape}.")

    return model, loss_history
