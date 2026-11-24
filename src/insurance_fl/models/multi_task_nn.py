"""Multi-task neural network for the 5 insurance prediction tasks.

CPU-friendly PyTorch model with a shared trunk and one head per task.
Implements the FL weight interface: ``get_weights()`` / ``set_weights()``.

Design decisions
----------------
* ``forward()`` returns **raw logits** (regression values / binary logits /
  multi-class logits). Activation functions are applied in :meth:`predict`
  only, so losses can use numerically stable built-ins
  (``BCEWithLogitsLoss``, ``CrossEntropyLoss``) instead of clamped logs.
* No mutable default arguments (``hidden_dims=None`` -> factory default).
* **Kendall uncertainty weighting** (Task 1): one learnable log-variance
  ``log_vars[i] = log sigma_i^2`` per task, zero-initialized so the
  Kendall loss starts out equal to the equal-weight sum (Layer 1
  continuity). ``log_vars`` are regular ``nn.Parameter``s and therefore
  part of ``state_dict`` -> they are federated by FedAvg like any other
  non-BN tensor (FedBN excludes BatchNorm only).
* **FedBN** (Task 1): BatchNorm tensors (weight/bias/running stats/
  ``num_batches_tracked``) are *local* to each client. They never cross
  the wire: the FL transport uses :meth:`get_federated_weights` /
  :meth:`set_federated_weights` which exclude exactly the BN keys, so
  the server-side FedAvg only ever averages non-BN parameters.
* Encoder widths follow the spec canon (``Task_Specification.md`` Task 1):
  3 dense layers 256 -> 128 -> 64.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

DEFAULT_TASK_TYPES: dict[str, str] = {
    "claim_risk_score": "regression",
    "customer_lifetime_value": "regression",
    "will_renew": "binary",
    "is_high_risk": "binary",
    "customer_segment": "multi",
}

DEFAULT_HIDDEN_DIMS: list[int] = [256, 128, 64]


class MultiTaskNN(nn.Module):
    """Multi-task network: shared trunk + per-task heads.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    task_types : Optional[Dict[str, str]]
        Mapping task -> "regression" | "binary" | "multi".
        Defaults to the 5 insurance tasks.
    hidden_dims : Optional[List[int]]
        Widths of shared hidden layers (default ``[256, 128, 64]``,
        the spec's encoder 256 -> 128 -> 64).
    dropout_rate : float
        Dropout probability (0 disables it).
    use_batch_norm : bool
        Insert BatchNorm1d after each shared linear layer (FedBN-local).
    num_classes : Optional[Dict[str, int]]
        Class counts for "multi" tasks (default 3 for customer_segment).
    """

    def __init__(
        self,
        input_dim: int,
        task_types: dict[str, str] | None = None,
        hidden_dims: list[int] | None = None,
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True,
        num_classes: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        hidden_dims = list(hidden_dims) if hidden_dims else list(DEFAULT_HIDDEN_DIMS)
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}")

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.task_types = dict(task_types) if task_types else dict(DEFAULT_TASK_TYPES)
        self.num_classes = dict(num_classes) if num_classes else {"customer_segment": 3}

        # --- Shared trunk ------------------------------------------------
        layers: list[nn.Module] = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim
        self.shared_backbone = nn.Sequential(*layers)
        last_dim = hidden_dims[-1] if hidden_dims else input_dim

        # --- Task heads ---------------------------------------------------
        self.heads = nn.ModuleDict()
        for task_name, task_type in self.task_types.items():
            if task_type in ("regression", "binary"):
                out_dim = 1
            elif task_type == "multi":
                out_dim = int(self.num_classes.get(task_name, 3))
            else:
                raise ValueError(f"Unsupported task_type: {task_type!r}")
            self.heads[task_name] = nn.Linear(last_dim, out_dim)

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()

        # --- Kendall homoscedastic uncertainty (Task 1) --------------------
        # s_i = log(sigma_i^2), one per task, in task_types order.
        # Zero-init: exp(-0)=1 and +0 => the Kendall loss starts out equal
        # to the Layer 1 equal-weight sum (smooth baseline continuity).
        self.log_vars = nn.Parameter(torch.zeros(len(self.task_types)))
        self._task_index: dict[str, int] = {name: i for i, name in enumerate(self.task_types)}

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass. Returns RAW logits per task (no activation)."""
        features = self.shared_backbone(x)
        outputs: dict[str, torch.Tensor] = {}
        for task_name, head in self.heads.items():
            raw = head(features)
            if self.task_types[task_name] == "multi":
                outputs[task_name] = raw  # (B, n_cls) logits
            else:
                outputs[task_name] = raw.squeeze(-1)  # (B,) regression/logit
        return outputs

    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Inference: applies task activation to produce usable outputs.

        regression -> raw value; binary -> probability; multi -> probability.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            preds: dict[str, torch.Tensor] = {}
            for task_name, out in logits.items():
                ttype = self.task_types[task_name]
                if ttype == "binary":
                    preds[task_name] = torch.sigmoid(out)
                elif ttype == "multi":
                    preds[task_name] = torch.softmax(out, dim=-1)
                else:
                    preds[task_name] = out
        return preds

    def get_losses(
        self,
        logits: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute per-task losses from raw logits (stable formulations)."""
        losses: dict[str, torch.Tensor] = {}
        for task_name, task_type in self.task_types.items():
            if task_name not in targets or task_name not in logits:
                continue
            out = logits[task_name]
            target = targets[task_name].to(out.device)
            if task_type == "regression":
                losses[task_name] = self.mse_loss(out, target.float())
            elif task_type == "binary":
                losses[task_name] = self.bce_loss(out, target.float())
            elif task_type == "multi":
                losses[task_name] = self.ce_loss(out, target.long())
        return losses

    def get_total_loss(
        self,
        losses: dict[str, torch.Tensor],
        loss_weights: dict[str, float] | None = None,
    ) -> torch.Tensor:
        """Weighted sum of per-task losses (equal weights by default)."""
        weights = loss_weights or dict.fromkeys(losses, 1.0)
        device = next(self.parameters()).device
        total = torch.zeros((), device=device)
        for task_name, loss_val in losses.items():
            total = total + float(weights.get(task_name, 1.0)) * loss_val
        return total

    def get_kendall_total_loss(
        self,
        losses: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Kendall et al. (2018) uncertainty-weighted multi-task loss.

        ``total = sum_i [ exp(-s_i) * L_i + s_i ]`` with the learnable
        log-variances ``s_i = self.log_vars[i]`` (``log sigma_i^2``).

        The ``exp(-s_i)`` factor down-weights high-variance tasks (the
        ~1e7-scale regression MSEs) while ``+ s_i`` is the log-normalizer
        that prevents the trivial ``s_i -> +inf`` solution. With
        ``log_vars = 0`` this reduces exactly to :meth:`get_total_loss`
        with equal weights.

        Tasks absent from ``losses`` contribute nothing (mirrors
        :meth:`get_total_loss` skipping behavior for missing targets).
        """
        device = next(self.parameters()).device
        total = torch.zeros((), device=device)
        for task_name, loss_val in losses.items():
            idx = self._task_index.get(task_name)
            if idx is None:
                continue
            log_var = self.log_vars[idx]
            total = total + torch.exp(-log_var) * loss_val + log_var
        return total

    def uncertainty_weights(self) -> dict[str, float]:
        """Effective per-task loss weights ``exp(-log_sigma^2)`` (Kendall)."""
        with torch.no_grad():
            return {
                name: float(torch.exp(-self.log_vars[idx]).item())
                for name, idx in self._task_index.items()
            }

    # ------------------------------------------------------------------
    # Federated learning weight interface
    # ------------------------------------------------------------------
    def get_weights(self) -> list[np.ndarray]:
        """Return model parameters as a deterministic list of numpy arrays.

        Order is stable (state_dict insertion order), so FedAvg-style
        weighted averaging of aligned lists is well-defined.
        """
        self.eval()
        return [t.detach().cpu().numpy().copy() for t in self.state_dict().values()]

    def set_weights(self, weights: list[np.ndarray]) -> None:
        """Load parameters from a list produced by :meth:`get_weights`."""
        state = self.state_dict()
        if len(weights) != len(state):
            raise ValueError(f"Weight count mismatch: got {len(weights)}, expected {len(state)}")
        new_state = {}
        for key, value in zip(state.keys(), weights):
            arr = np.asarray(value)
            if tuple(arr.shape) != tuple(state[key].shape):
                raise ValueError(
                    f"Shape mismatch for {key}: got {arr.shape}, "
                    f"expected {tuple(state[key].shape)}"
                )
            new_state[key] = torch.as_tensor(arr, dtype=state[key].dtype)
        self.load_state_dict(new_state)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # FedBN: BatchNorm stays local (Task 1)
    # ------------------------------------------------------------------
    def local_bn_keys(self) -> list[str]:
        """``state_dict`` keys owned by BatchNorm layers (never leave the client).

        Empty when ``use_batch_norm=False`` (then federated == full weights).
        """
        prefixes = tuple(
            f"{name}."
            for name, module in self.named_modules()
            if isinstance(module, nn.BatchNorm1d)
        )
        return [key for key in self.state_dict() if key.startswith(prefixes)]

    def federated_keys(self) -> list[str]:
        """``state_dict`` keys that cross the wire (everything except BN)."""
        bn_keys = set(self.local_bn_keys())
        return [key for key in self.state_dict() if key not in bn_keys]

    def get_federated_weights(self) -> list[np.ndarray]:
        """Non-BN parameters in ``state_dict`` order (the FedBN wire format).

        Same determinism as :meth:`get_weights`: identical architectures
        produce identical key order, so FedAvg's list-wise averaging is
        well-defined across clients.
        """
        self.eval()
        state = self.state_dict()
        return [state[key].detach().cpu().numpy().copy() for key in self.federated_keys()]

    def set_federated_weights(self, weights: list[np.ndarray]) -> None:
        """Load non-BN parameters; local BatchNorm state is left untouched."""
        keys = self.federated_keys()
        if len(weights) != len(keys):
            raise ValueError(
                f"Federated weight count mismatch: got {len(weights)}, "
                f"expected {len(keys)} (state_dict minus BatchNorm)"
            )
        state = self.state_dict()
        new_state = {}
        for key, value in zip(keys, weights):
            arr = np.asarray(value)
            if tuple(arr.shape) != tuple(state[key].shape):
                raise ValueError(
                    f"Shape mismatch for {key}: got {arr.shape}, "
                    f"expected {tuple(state[key].shape)}"
                )
            new_state[key] = torch.as_tensor(arr, dtype=state[key].dtype)
        # strict=False: only the provided (non-BN) keys are updated.
        self.load_state_dict(new_state, strict=False)

    def get_local_bn_state(self) -> dict[str, np.ndarray]:
        """Snapshot of local BatchNorm tensors (for ``context.state`` storage)."""
        state = self.state_dict()
        return {key: state[key].detach().cpu().numpy().copy() for key in self.local_bn_keys()}

    def set_local_bn_state(self, bn: dict[str, np.ndarray]) -> None:
        """Restore a snapshot produced by :meth:`get_local_bn_state`.

        Keys must match the model's current BN keys exactly (same
        architecture is guaranteed within a run — clients rebuild the
        model per message with identical construction arguments).
        """
        expected = self.local_bn_keys()
        if set(bn) != set(expected):
            raise ValueError(
                f"BN state key mismatch: got {sorted(bn)}, expected {sorted(expected)}"
            )
        state = self.state_dict()
        new_state = {}
        for key in expected:
            arr = np.asarray(bn[key])
            if tuple(arr.shape) != tuple(state[key].shape):
                raise ValueError(
                    f"Shape mismatch for {key}: got {arr.shape}, "
                    f"expected {tuple(state[key].shape)}"
                )
            new_state[key] = torch.as_tensor(arr, dtype=state[key].dtype)
        self.load_state_dict(new_state, strict=False)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Save network state and construction metadata to ``path``."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "model.pt")
        meta = {
            "input_dim": self.input_dim,
            "task_types": self.task_types,
            "hidden_dims": self.hidden_dims,
            "dropout_rate": self.dropout_rate,
            "use_batch_norm": self.use_batch_norm,
            "num_classes": self.num_classes,
        }
        (path / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> MultiTaskNN:
        """Load a model saved by :meth:`save`."""
        path = Path(path)
        meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        model = cls(
            input_dim=meta["input_dim"],
            task_types=meta["task_types"],
            hidden_dims=meta["hidden_dims"],
            dropout_rate=meta["dropout_rate"],
            use_batch_norm=meta["use_batch_norm"],
            num_classes=meta["num_classes"],
        )
        state = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        return model
