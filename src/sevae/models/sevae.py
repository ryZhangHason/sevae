from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Config
# -----------------------------
@dataclass
class SEVAEConfig:
    # Core structure
    n_constructs: int                       # K
    items_per_construct: Union[int, List[int]]  # J or [J1, J2, ...]
    d_per_construct: int = 1                # latent per construct (your d_lat)
    # Nuisance ("method") latents: one or more small blocks over full x
    d_nuisance: int = 1
    n_nuisance_blocks: int = 1

    # Capacity / regularization
    hidden: int = 32
    dropout: float = 0.05
    context_dim: int = 1                    # global cross-talk (like your global_context)

    # Loss weights (you can anneal kl_weight externally each epoch)
    kl_weight: float = 1.0                  # β
    tc_weight: float = 0.0                  # γ
    ortho_weight: float = 0.0               # α
    leakage_weight: float = 0.0             # λ_adv

    # Behavioural switches to match your script
    tc_on_construct_only: bool = True       # TC on z_c only (like your tc_loss(zs))
    adv_include_block_recon: bool = True    # include per-group recon inside adv loss (your adversarial_leakage_loss)

    # Reconstruction reduction
    recon_reduction: str = "mean"           # "mean" or "sum" (your script used "sum")
    adv_recon_reduction: str = "mean"       # reduction for adversarial and block recon terms

    # ---- Flexible column indexing ----
    # If you pass names at runtime (bind_feature_names), set feature_name_groups in config.
    # If you want to pass integer indices directly, call bind_column_groups([...]) before training.
    feature_name_groups: Optional[List[List[str]]] = None   # per-construct lists of column names
    # NOTE: If neither is provided, we assume contiguous blocks via items_per_construct.

# -----------------------------
# Model
# -----------------------------
class SEVAE(nn.Module):
    """
    SE-VAE for tabular constructs with:
      - Global cross-talk context (context_dim features per construct)
      - K construct encoders: each sees its item block + its context chunk
      - N nuisance ("method") encoders over full x (concatenated to zn)
      - K decoders: each reconstructs its own item block from [z_k, zn]
      - K adversarial decoders: each tries to reconstruct its block from zn alone

    Flexible column indexing:
      - Default: contiguous blocks with items_per_construct (int or list[int])
      - Arbitrary index groups: call `bind_column_groups(list[list[int]])`
      - Name-based groups: set cfg.feature_name_groups & call `bind_feature_names(names)`
    """

    def __init__(self, cfg: SEVAEConfig):
        super().__init__()
        self.cfg = cfg

        # Will be lazily initialized on first forward (need input dim)
        self._D_total: Optional[int] = None
        self.items_per_construct_list: Optional[List[int]] = None
        self.group_indices: Optional[List[torch.LongTensor]] = None  # per-construct column indices
        self._uses_gather: bool = False  # True if indices are non-contiguous

        # Modules that need D will be built lazily.
        self.global_context: Optional[nn.Sequential] = None
        self.factor_encoders: nn.ModuleList = nn.ModuleList()
        self.nuisance_encoders: nn.ModuleList = nn.ModuleList()
        self.decoders: nn.ModuleList = nn.ModuleList()
        self.adversarial_decoders: nn.ModuleList = nn.ModuleList()

    # ---------- Public API for flexible indexing ----------

    def bind_feature_names(self, feature_names: Sequence[str]) -> None:
        """
        Provide the original column names once, so we can map cfg.feature_name_groups to indices.
        Call this before the first forward if you're using name-based groups.
        """
        if self.cfg.feature_name_groups is None:
            raise ValueError("cfg.feature_name_groups is None; nothing to bind. "
                             "Either set feature_name_groups in config or call bind_column_groups(...).")
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        groups: List[List[int]] = []
        for g, names in enumerate(self.cfg.feature_name_groups):
            idxs = []
            for nm in names:
                if nm not in name_to_idx:
                    raise KeyError(f"Feature name '{nm}' (construct {g}) not found in provided feature_names.")
                idxs.append(name_to_idx[nm])
            groups.append(idxs)
        self.bind_column_groups(groups)

    def bind_column_groups(self, column_groups: List[List[int]]) -> None:
        """
        Provide explicit column indices for each construct.
        """
        if len(column_groups) != self.cfg.n_constructs:
            raise ValueError(f"column_groups must have length {self.cfg.n_constructs}")
        # Store as tensors (sorted for stability but order doesn't matter)
        self.group_indices = [torch.as_tensor(sorted(g), dtype=torch.long) for g in column_groups]
        self.items_per_construct_list = [len(g) for g in column_groups]
        self._uses_gather = True

    # ---------- Lazy init / building modules ----------

    def _infer_items_per_construct_list(self, D: int) -> List[int]:
        if self.items_per_construct_list is not None:
            return self.items_per_construct_list
        ipc = self.cfg.items_per_construct
        if isinstance(ipc, int):
            total = self.cfg.n_constructs * ipc
            if total != D:
                raise ValueError(f"Expected D={total} from n_constructs*items_per_construct, got D={D}")
            return [ipc] * self.cfg.n_constructs
        else:
            if sum(ipc) != D or len(ipc) != self.cfg.n_constructs:
                raise ValueError(f"items_per_construct list must sum to D and have length n_constructs. "
                                 f"Got sum={sum(ipc)}, len={len(ipc)}, D={D}, K={self.cfg.n_constructs}")
            return list(ipc)

    def _init_groups_if_needed(self, D: int) -> None:
        if self._D_total is not None:
            return
        self._D_total = D

        # If user has not bound indices, assume contiguous blocks
        if self.group_indices is None:
            items = self._infer_items_per_construct_list(D)
            self.items_per_construct_list = items
            # build contiguous indices
            offs = 0
            self.group_indices = []
            for j in items:
                idx = torch.arange(offs, offs + j, dtype=torch.long)
                self.group_indices.append(idx)
                offs += j
            self._uses_gather = False  # contiguous slices
        else:
            # already provided via bind_column_groups
            # sanity check sum of lengths == D
            if sum(len(g) for g in self.group_indices) != D:
                raise ValueError("Sum of lengths in column_groups does not equal input dimension D.")

        # Build modules now that D is known
        self._build_modules(D)

    def _build_modules(self, D: int) -> None:
        K = self.cfg.n_constructs
        J_list = self.items_per_construct_list
        assert J_list is not None
        d_lat = self.cfg.d_per_construct
        d_n = self.cfg.d_nuisance
        nnb = self.cfg.n_nuisance_blocks
        H = self.cfg.hidden
        p = self.cfg.dropout
        ctx = self.cfg.context_dim

        # Global context over full x
        if ctx > 0:
            self.global_context = nn.Sequential(
                nn.Linear(D, H),
                nn.ReLU(),
                nn.Dropout(p),
                nn.Linear(H, K * ctx),
            )
        else:
            self.global_context = None

        # Per-construct encoders (group items + context_chunk -> 2*d_lat)
        self.factor_encoders = nn.ModuleList()
        in_sizes = [j + (ctx if ctx > 0 else 0) for j in J_list]
        for ins in in_sizes:
            self.factor_encoders.append(
                nn.Sequential(
                    nn.Linear(ins, H),
                    nn.ReLU(),
                    nn.Dropout(p),
                    nn.Linear(H, 2 * d_lat),
                )
            )

        # Nuisance encoders over full x
        self.nuisance_encoders = nn.ModuleList()
        for _ in range(nnb):
            self.nuisance_encoders.append(
                nn.Sequential(
                    nn.Linear(D, H),
                    nn.ReLU(),
                    nn.Dropout(p),
                    nn.Linear(H, 2 * d_n),
                )
            )

        # Per-construct decoders: [z_k, zn] -> items_k
        self.decoders = nn.ModuleList()
        dec_in = d_lat + (d_n * nnb if nnb > 0 else 0)
        for j in J_list:
            self.decoders.append(
                nn.Sequential(
                    nn.Linear(dec_in, H),
                    nn.ReLU(),
                    nn.Dropout(p),
                    nn.Linear(H, j),
                )
            )

        # Adversarial decoders: zn -> items_k
        self.adversarial_decoders = nn.ModuleList()
        if nnb > 0:
            adv_in = d_n * nnb
            adv_h = max(2, H // 2)
            for j in J_list:
                self.adversarial_decoders.append(
                    nn.Sequential(
                        nn.Linear(adv_in, adv_h),
                        nn.ReLU(),
                        nn.Linear(adv_h, j),
                    )
                )
        else:
            # placeholder empty list
            self.adversarial_decoders = nn.ModuleList([nn.Identity() for _ in J_list])

    # ---------- Helpers ----------

    def _split_groups(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Split x into K tensors according to group_indices.
        Works with contiguous slices or arbitrary index lists (via gather).
        """
        assert self.group_indices is not None
        groups = []
        if self._uses_gather:
            for idx in self.group_indices:
                groups.append(x.index_select(dim=1, index=idx))
        else:
            # fast slicing
            off = 0
            for j in self.items_per_construct_list or []:
                groups.append(x[:, off:off + j])
                off += j
        return groups

    @staticmethod
    def _kl_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if mu is None or logvar is None:
            return torch.zeros((), device=mu.device if isinstance(mu, torch.Tensor) else "cpu")
        return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()

    @staticmethod
    def _tc_penalty(z: torch.Tensor) -> torch.Tensor:
        # squared off-diagonal covariance energy (like your tc_loss)
        zc = z - z.mean(0, keepdim=True)
        cov = (zc.T @ zc) / (z.size(0) - 1 + 1e-9)
        off = cov - torch.diag(torch.diag(cov))
        return (off ** 2).sum()

    @staticmethod
    def _ortho_penalty(z: torch.Tensor) -> torch.Tensor:
        # squared off-diagonal of Gram/correlation (like your orthogonality_loss)
        g = (z.T @ z) / (z.size(0) + 1e-9)
        off = g - torch.diag(torch.diag(g))
        return (off ** 2).sum()

    # ---------- Encode / Decode ----------

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Returns:
          zc_mu [B, K*d], zc_logvar [B, K*d], zn_mu [B, nnb*d_n] or None, zn_logvar [B, nnb*d_n] or None
        """
        B, D = x.shape
        self._init_groups_if_needed(D)

        # Global context
        if self.global_context is not None:
            ctx_all = self.global_context(x)  # [B, K*ctx]
            ctx_chunks = torch.chunk(ctx_all, self.cfg.n_constructs, dim=1)  # list of [B, ctx]
        else:
            ctx_chunks = [None] * self.cfg.n_constructs

        # Group inputs
        x_groups = self._split_groups(x)

        # Construct encoders
        zc_mu, zc_logvar = [], []
        d = self.cfg.d_per_construct
        for k in range(self.cfg.n_constructs):
            xg = x_groups[k]
            if ctx_chunks[k] is not None:
                x_in = torch.cat([xg, ctx_chunks[k]], dim=1)
            else:
                x_in = xg
            h = self.factor_encoders[k](x_in)
            zc_mu.append(h[:, :d])
            zc_logvar.append(h[:, d:])
        zc_mu = torch.cat(zc_mu, dim=1)         # [B, K*d]
        zc_logvar = torch.cat(zc_logvar, dim=1) # [B, K*d]

        # Nuisance encoders (full x)
        if self.cfg.n_nuisance_blocks > 0:
            zn_mus, zn_logvars = [], []
            for enc in self.nuisance_encoders:
                h = enc(x)
                zn_mus.append(h[:, : self.cfg.d_nuisance])
                zn_logvars.append(h[:, self.cfg.d_nuisance :])
            zn_mu = torch.cat(zn_mus, dim=1)      # [B, nnb*d_n]
            zn_logvar = torch.cat(zn_logvars, dim=1)
        else:
            zn_mu = None
            zn_logvar = None

        return zc_mu, zc_logvar, zn_mu, zn_logvar

    @staticmethod
    def _reparam(mu: Optional[torch.Tensor], logvar: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if mu is None or logvar is None:
            return None
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, zc: torch.Tensor, zn: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Decode each construct block from [z_k, zn], concatenate along feature axis.
        """
        outs = []
        d = self.cfg.d_per_construct
        for k, dec in enumerate(self.decoders):
            z_k = zc[:, k * d : (k + 1) * d]
            if zn is not None:
                z_in = torch.cat([z_k, zn], dim=1)
            else:
                z_in = z_k
            x_hat_k = dec(z_in)
            outs.append(x_hat_k)
        return torch.cat(outs, dim=1)

    # ---------- Forward / Loss ----------

    def forward(self, x: torch.Tensor) -> dict:
        zc_mu, zc_logvar, zn_mu, zn_logvar = self.encode(x)
        zc = self._reparam(zc_mu, zc_logvar)
        zn = self._reparam(zn_mu, zn_logvar) if zn_mu is not None else None
        x_hat = self.decode(zc, zn)
        return {
            "x_hat": x_hat,
            "zc_mu": zc_mu, "zc_logvar": zc_logvar, "zc": zc,
            "zn_mu": zn_mu, "zn_logvar": zn_logvar, "zn": zn,
        }

    def loss(self, x: torch.Tensor, out: dict) -> dict:
        """
        Matches your script semantics:
          total =
              recon
            + kl_weight * (KL(zc) + KL(zn))
            + tc_weight * TC(zc)     (or TC([zc,zn]) if tc_on_construct_only=False)
            + ortho_weight * ORTHO(zc)
            + leakage_weight * (adv_recon(zn→x_k) [+ avg_group_recon([z_k, zn]→x_k)])
        """
        x_hat = out["x_hat"]
        zc_mu, zc_logvar, zc = out["zc_mu"], out["zc_logvar"], out["zc"]
        zn_mu, zn_logvar, zn = out.get("zn_mu", None), out.get("zn_logvar", None), out.get("zn", None)

        # main recon (whole vector)
        if self.cfg.recon_reduction == "sum":
            loss_recon = F.mse_loss(x_hat, x, reduction="sum") / x.size(0)
        else:
            loss_recon = F.mse_loss(x_hat, x, reduction="mean")

        # KL terms
        loss_kl_construct = self._kl_normal(zc_mu, zc_logvar)
        loss_kl_nuisance = self._kl_normal(zn_mu, zn_logvar) if zn_mu is not None else torch.zeros_like(loss_recon)

        # TC & Ortho
        tc_input = zc if (self.cfg.tc_on_construct_only or zn is None) else torch.cat([zc, zn], dim=1)
        loss_tc = self._tc_penalty(tc_input) if self.cfg.tc_weight > 0 else torch.zeros_like(loss_recon)
        loss_ortho = self._ortho_penalty(zc) if self.cfg.ortho_weight > 0 else torch.zeros_like(loss_recon)

        # Leakage: adversarial (zn → x_k)
        if self.cfg.leakage_weight > 0 and zn is not None and len(self.adversarial_decoders) > 0:
            x_groups = self._split_groups(x)
            per_k = []
            for k, adv in enumerate(self.adversarial_decoders):
                x_hat_k_adv = adv(zn.detach())
                if self.cfg.adv_recon_reduction == "sum":
                    r = F.mse_loss(x_hat_k_adv, x_groups[k], reduction="sum") / x.size(0)
                else:
                    r = F.mse_loss(x_hat_k_adv, x_groups[k], reduction="mean")
                per_k.append(r)
            loss_leakage = torch.stack(per_k).mean()
        else:
            loss_leakage = torch.zeros_like(loss_recon)

        # Optional: add per-group recon ([z_k, zn] → x_k) inside the adversarial bucket, like your script
        if self.cfg.adv_include_block_recon:
            x_groups = self._split_groups(x)
            d = self.cfg.d_per_construct
            per_k = []
            for k, dec in enumerate(self.decoders):
                z_k = zc[:, k * d : (k + 1) * d]
                z_in = torch.cat([z_k, zn], dim=1) if zn is not None else z_k
                x_hat_k = dec(z_in)
                if self.cfg.adv_recon_reduction == "sum":
                    r = F.mse_loss(x_hat_k, x_groups[k], reduction="sum") / x.size(0)
                else:
                    r = F.mse_loss(x_hat_k, x_groups[k], reduction="mean")
                per_k.append(r)
            block_recon = torch.stack(per_k).mean()
        else:
            block_recon = torch.zeros_like(loss_recon)

        loss_total = (
            loss_recon
            + self.cfg.kl_weight * (loss_kl_construct + loss_kl_nuisance)
            + self.cfg.tc_weight * loss_tc
            + self.cfg.ortho_weight * loss_ortho
            + self.cfg.leakage_weight * (loss_leakage + block_recon)
        )

        return {
            "loss_total": loss_total,
            "loss_recon": loss_recon,
            "loss_kl_construct": loss_kl_construct,
            "loss_kl_nuisance": loss_kl_nuisance,
            "loss_tc": loss_tc,
            "loss_ortho": loss_ortho,
            "loss_leakage": loss_leakage,
            "loss_block_recon": block_recon,
        }