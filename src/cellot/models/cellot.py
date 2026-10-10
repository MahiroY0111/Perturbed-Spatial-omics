from pathlib import Path
import torch
from collections import namedtuple
from cellot.networks.icnns import ICNN

from absl import flags

FLAGS = flags.FLAGS

FGPair = namedtuple("FGPair", "f g")


def load_networks(config, **kwargs):
    def unpack_kernel_init_fxn(name="uniform", **kwargs):
        if name == "normal":

            def init(*args):
                return torch.nn.init.normal_(*args, **kwargs)

        elif name == "uniform":

            def init(*args):
                return torch.nn.init.uniform_(*args, **kwargs)

        else:
            raise ValueError

        return init

    kwargs.setdefault("hidden_units", [64] * 4)
    kwargs.update(dict(config.get("model", {})))

    # eg parameters specific to g are stored in config.model.g
    kwargs.pop("name")
    if "latent_dim" in kwargs:
        kwargs.pop("latent_dim")
    fupd = kwargs.pop("f", {})
    gupd = kwargs.pop("g", {})

    fkwargs = kwargs.copy()
    fkwargs.update(fupd)
    fkwargs["kernel_init_fxn"] = unpack_kernel_init_fxn(
        **fkwargs.pop("kernel_init_fxn")
    )

    gkwargs = kwargs.copy()
    gkwargs.update(gupd)
    gkwargs["kernel_init_fxn"] = unpack_kernel_init_fxn(
        **gkwargs.pop("kernel_init_fxn")
    )

    f = ICNN(**fkwargs)
    g = ICNN(**gkwargs)

    if "verbose" in FLAGS and FLAGS.verbose:
        print(g)
        print(kwargs)

    return f, g


def load_opts(config, f, g):
    kwargs = dict(config.get("optim", {}))
    assert kwargs.pop("optimizer", "Adam") == "Adam"

    fupd = kwargs.pop("f", {})
    gupd = kwargs.pop("g", {})

    fkwargs = kwargs.copy()
    fkwargs.update(fupd)
    fkwargs["betas"] = (fkwargs.pop("beta1", 0.9), fkwargs.pop("beta2", 0.999))

    gkwargs = kwargs.copy()
    gkwargs.update(gupd)
    gkwargs["betas"] = (gkwargs.pop("beta1", 0.9), gkwargs.pop("beta2", 0.999))

    opts = FGPair(
        f=torch.optim.Adam(f.parameters(), **fkwargs),
        g=torch.optim.Adam(g.parameters(), **gkwargs),
    )

    return opts


def load_cellot_model(config, restore=None, **kwargs):
    f, g = load_networks(config, **kwargs)
    opts = load_opts(config, f, g)

    if restore is not None and Path(restore).exists():
        ckpt = torch.load(restore)
        f.load_state_dict(ckpt["f_state"])
        opts.f.load_state_dict(ckpt["opt_f_state"])

        g.load_state_dict(ckpt["g_state"])
        opts.g.load_state_dict(ckpt["opt_g_state"])

    return (f, g), opts


# transport = g.transport(source) が T(x)
# f(transport) が f(T(x))
# torch.multiply(source, transport).sum(-1) が ⟨x, T(x)⟩

#ニューラルネットワークの学習のための損失関数（ｇについて）
def compute_loss_g(f, g, source, transport=None):
    if transport is None:
        transport = g.transport(source)
    #print(':compute_loss_g:torch.multiply',torch.multiply(source, transport).sum(-1, keepdim=True))
    #print(':compute_loss_g:f(transport)',f(transport))
    return f(transport) - torch.multiply(source, transport).sum(-1, keepdim=True)



# def compute_loss_g_zonation?_ver(f,
#                    g,
#                    source,          # ctrl latent: [B, D]
#                    h_zon,          # 上で学習した zonation regressor
#                    lambda_zon=0.1,
#                    transport=None):
#     """
#     Neural OT の g 用 loss + zonation 正則化
#     """
#     if transport is None:
#         transport = g.transport(source)   # [B, D]

#     # もとの Neural OT 部分
#     base = f(transport) - torch.multiply(source, transport).sum(-1, keepdim=True)
#     # base: [B,1]

#     # zonation スコア（source 側は勾配いらないので no_grad）
#     with torch.no_grad():
#         s_src = h_zon(source)         # [B,1]
#     s_tr  = h_zon(transport)          # [B,1]  ← ここを通じて g に勾配が流れる

#     zon_loss = (s_tr - s_src).pow(2)  # [B,1]
#     zon_loss = zon_loss               # バッチ平均は外側で .mean() でもいいしここで取ってもOK

#     return base + lambda_zon * zon_loss



def compute_g_constraint(g, form=None, beta=0):
    if form is None or form == "None":
        return 0

    if form == "clamp":
        g.clamp_w()
        return 0

    elif form == "fnorm":
        if beta == 0:
            return 0

        return beta * sum(map(lambda w: w.weight.norm(p="fro"), g.W))

    raise ValueError

#ニューラルネットワークの学習のための損失関数（ｆについて）
def compute_loss_f(f, g, source, target, transport=None):
    if transport is None:
        transport = g.transport(source)

    return -f(transport) + f(target)



# 空間的整合性のロス関数
def compute_spatial_consistency_loss(B, W, Y_hat, d=None):
    """
    B: torch.Tensor, shape (n_spots, n_cells)
    W: torch.Tensor, shape (n_cells, n_types)
    Y_hat: torch.Tensor, shape (n_cells, n_genes)  # Tθ(X)のデコード結果
    d: torch.Tensor or None, shape (n_cells, 1) or (n_cells,)  # オプションの重み
    """
    # E_cell_post = Norm(W^T Y_hat ⊙ d)
    E_cell = torch.matmul(W.T, Y_hat)  # (n_types, n_genes)
    if d is not None:
        E_cell = E_cell * d.view(-1, 1)  # broadcasting

    # E_spot_post = Norm(B^T Y_hat)
    E_spot = torch.matmul(B.T, Y_hat)  # (n_spots, n_genes)

    # Norm: L1ノルム or L2ノルム（論文に合わせて選択）
    # ここではL1ノルム
    loss = torch.nn.functional.l1_loss(E_spot, E_cell)
    return loss

#評価用でこの二乗ワッサーシュタイン距離はニューラルネットの学習の損失関数には使用しない
def compute_w2_distance(f, g, source, target, transport=None):
    if transport is None:
        transport = g.transport(source).squeeze()

    with torch.no_grad():
        Cpq = (source * source).sum(1, keepdim=True) + (target * target).sum(
            1, keepdim=True
        )
        Cpq = 0.5 * Cpq

        cost = (
            f(transport)
            - torch.multiply(source, transport).sum(-1, keepdim=True)
            - f(target)
            + Cpq
        )
        cost = cost.mean()
    return cost


def numerical_gradient(param, fxn, *args, eps=1e-4):
    with torch.no_grad():
        param += eps
    plus = float(fxn(*args))

    with torch.no_grad():
        param -= 2 * eps
    minus = float(fxn(*args))

    with torch.no_grad():
        param += eps

    return (plus - minus) / (2 * eps)
