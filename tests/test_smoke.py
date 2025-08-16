import torch
from sevae import SEVAE, SEVAEConfig

def test_forward_smoke():
    cfg = SEVAEConfig(
        n_constructs=3, items_per_construct=4,
        d_per_construct=1, d_nuisance=2, n_nuisance_blocks=1,
        hidden=32, context_dim=8, dropout=0.0,
        tc_weight=0.0, ortho_weight=0.0, leakage_weight=0.0
    )
    model = SEVAE(cfg)
    x = torch.randn(5, 3*4)
    out = model(x)
    losses = model.loss(x, out)
    assert out["x_hat"].shape == x.shape
    assert torch.isfinite(losses["loss_total"]).item()