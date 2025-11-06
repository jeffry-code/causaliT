
"""
Template: optimize over P using a trained proT model checkpoint.

You must implement two small functions for YOUR setup:
 - build_input(P): turn a numpy vector of controllable parameters into the model's input tensor
 - run_model(x): run the loaded model and return a scalar prediction Y_hat

Why this design? Your repo's proT training loop and embedding expect specific shapes and maps.
Keeping this modular lets you adapt quickly without touching the optimizer code.
"""
from __future__ import annotations
import os
import json
import numpy as np

import torch
from torch import nn

from predictors import ModelPredictor
from objectives import Objective
from optimizers import cma_es, adam


# ---------------------- USER TODO: fill these for your dataset ----------------------
# Load the variable maps produced by SCMDataset.generate_ds(...)
DATA_DIR = "data/example"  # change to your dataset folder
with open(os.path.join(DATA_DIR, "input_vars_map.json"), "r") as f:
    iv_map = json.load(f)  # e.g., {"P1": 1, "P2": 2, ...}
with open(os.path.join(DATA_DIR, "target_vars_map.json"), "r") as f:
    tv_map = json.load(f)  # e.g., {"Y": 1}

# Choose which variables are controllable (order defines P vector)
CONTROLLABLE = ["P1","P2","P3","P4","P5"]  # edit to ["P1","P2","P3","P4","P5"] later

# Bounds in same order as CONTROLLABLE
LB = np.array([-3]*5, dtype=float)
UB = np.array([3]*5, dtype=float)
# Path to your trained checkpoint & model load function
CKPT_PATH = "experiments/example/k_0/checkpoints/best_checkpoint.ckpt"  # <- update to your checkpoint

# You need to provide two functions:
from proT.training.forecasters.transformer_forecaster import TransformerForecaster

def _load_checkpoint_cpu(ckpt_path: str) -> TransformerForecaster:
    # Load checkpoint on CPU and force CPU device in the saved config
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        # For older torch without weights_only, fall back to classic load
        ckpt = torch.load(ckpt_path, map_location="cpu")
    config = ckpt["hyper_parameters"]
    try:
        if "model" in config and "kwargs" in config["model"]:
            config["model"]["kwargs"]["device"] = "cpu"
    except Exception:
        pass
    model = TransformerForecaster(config)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model

# Replace eager Lightning load with CPU-safe loader
# _LIT_MODEL = TransformerForecaster.load_from_checkpoint(CKPT_PATH).eval()
_LIT_MODEL = _load_checkpoint_cpu(CKPT_PATH).eval()
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_LIT_MODEL = _LIT_MODEL.to(_DEVICE)

# Load LightningModule once

def load_model(ckpt_path: str) -> nn.Module:
    """
    Load your proT model wrapper (LightningModule).
    Kept for API parity; the global _LIT_MODEL is already loaded.
    """
    return _LIT_MODEL


def build_input(P: np.ndarray) -> torch.Tensor:
    """
    Convert controllable vector P into the (B=1, L_in, D=2) input expected by the encoder.
    Strategy:
      - Build a full variable list in the same order used during training.
      - For controllable Ps: put provided values.
      - For non-controllable inputs (e.g., C*), set to NaN; the encoder mask should handle missings.
    """
    # Load sequential order used during training (iv_order is stored in enc mask CSV header order)
    # Here we reconstruct minimally using the iv_map's key order sorted by index
    inv = sorted(((idx, var) for var, idx in iv_map.items()), key=lambda t: t[0])
    seq_vars = [var for _, var in inv]  # e.g., ["P1","P2","P3","P4","P5","C1",...]
    # Map controllables to values
    ctrl_vals = {var: P[i] for i, var in enumerate(CONTROLLABLE)}
    values = []
    var_ids = []
    for var in seq_vars:
        if var in ctrl_vals:
            v = float(ctrl_vals[var])
        else:
            v = float("nan")  # let mask handle missing
        values.append(v)
        var_ids.append(float(iv_map[var]))
    arr = np.stack([values, var_ids], axis=-1)[None, ...]  # shape (1, L_in, 2)
    return torch.tensor(arr, dtype=torch.float32)

def build_target() -> torch.Tensor:
    """
    Build decoder input (B=1, L_dec, D=2) with zeroed values and correct variable ids.
    This avoids masking out the decoder and matches the training wrapper expectations.
    """
    inv = sorted(((idx, var) for var, idx in tv_map.items()), key=lambda t: t[0])
    seq_vars = [var for _, var in inv]  # typically ["Y"]
    values = [0.0 for _ in seq_vars]
    var_ids = [float(tv_map[var]) for var in seq_vars]
    arr = np.stack([values, var_ids], axis=-1)[None, ...]  # shape (1, L_dec, 2)
    return torch.tensor(arr, dtype=torch.float32)


def run_model(x: torch.Tensor) -> torch.Tensor:
    """
    Run the loaded Lightning wrapper and return a scalar prediction tensor Y_hat for batch size 1.
    Supplies both encoder input (x) and decoder input (y) to the model.
    """
    y = build_target().to(x.device)
    model = load_model(CKPT_PATH).eval().to(x.device)
    with torch.no_grad():
        forecast_out, *_ = model.forward(data_input=x, data_trg=y)
    # ensure a (1, 1) scalar prediction tensor
    y_hat = forecast_out
    if y_hat.ndim == 1:
        y_hat = y_hat.view(1, 1)
    elif y_hat.ndim >= 2:
        y_hat = y_hat.view(1, 1)
    return y_hat


# ---------------------- Gradient-based optimization (autograd) ----------------------

def build_input_torch(P: torch.Tensor) -> torch.Tensor:
    """
    Differentiable builder: map P (requires_grad) into encoder input tensor (1, L_in, 2).
    Non-controllables are set to NaN (handled by embeddings via nan_to_num), so gradients
    flow only through controllable entries.
    """
    inv = sorted(((idx, var) for var, idx in iv_map.items()), key=lambda t: t[0])
    seq_vars = [var for _, var in inv]
    values: list[torch.Tensor] = []
    var_ids: list[torch.Tensor] = []
    ctrl_vals = {var: P[i] for i, var in enumerate(CONTROLLABLE)}
    for var in seq_vars:
        if var in ctrl_vals:
            v = ctrl_vals[var]
            if not torch.is_tensor(v):
                v = torch.tensor(float(v), dtype=torch.float32, device=_DEVICE)
        else:
            v = torch.tensor(float("nan"), dtype=torch.float32, device=_DEVICE)
        values.append(v)
        var_ids.append(torch.tensor(float(iv_map[var]), dtype=torch.float32, device=_DEVICE))
    arr = torch.stack([torch.stack(values), torch.stack(var_ids)], dim=-1).unsqueeze(0)
    return arr  # (1, L_in, 2)


def objective_torch(P_vec: torch.Tensor) -> torch.Tensor:
    """
    Maximize predicted Y by minimizing -Y_hat.
    """
    X = build_input_torch(P_vec.to(_DEVICE))
    Y = build_target().to(_DEVICE)
    forecast_out, *_ = _LIT_MODEL.forward(data_input=X, data_trg=Y)
    y_hat = forecast_out.reshape(-1)[0]
    return -y_hat


def torch_adam_opt(x0: np.ndarray, lb: np.ndarray, ub: np.ndarray, lr: float = 0.05, iters: int = 200):
    P = torch.nn.Parameter(torch.tensor(x0, dtype=torch.float32, device=_DEVICE))
    opt = torch.optim.Adam([P], lr=lr)
    best_f = float("inf")
    best_x = None
    history = []
    lb_t = torch.tensor(lb, dtype=torch.float32, device=_DEVICE)
    ub_t = torch.tensor(ub, dtype=torch.float32, device=_DEVICE)
    for t in range(1, iters + 1):
        opt.zero_grad(set_to_none=True)
        loss = objective_torch(P)
        loss.backward()
        opt.step()
        # Project to bounds
        with torch.no_grad():
            P.data = torch.max(torch.min(P.data, ub_t), lb_t)
        f = float(loss.item())
        if f < best_f:
            best_f = f
            best_x = P.detach().cpu().numpy().copy()
        history.append((best_f, float(torch.linalg.norm(P.detach()).item())))
    return best_x, best_f, history


def main_autograd():
    dim = len(CONTROLLABLE)
    x0 = np.zeros(dim, dtype=float)
    lb, ub = LB, UB
    x_best, f_best, hist = torch_adam_opt(x0=x0, lb=lb, ub=ub, lr=0.05, iters=200)
    print("[Torch-Adam] best f (=-Y_hat):", f_best, "x*:", x_best)
# ------------------------------------------------------------------------------------


def main():
    predictor = ModelPredictor(build_input=build_input, run_model=run_model)
    # We want to MAXIMIZE Y_hat
    objective = Objective(predictor, maximize=True)

    # init
    dim = len(CONTROLLABLE)
    x0 = np.zeros(dim, dtype=float)
    bounds = (LB, UB)

    # Try CMA-ES first
    res = cma_es(objective, x0=x0, sigma0=1.0, bounds=bounds, max_iters=200)
    print("[CMA-ES] f_best:", res.f_best, "x*:", res.x_best)

    # Optionally try Adam with finite differences
    res_adam = adam(objective, x0=x0, bounds=bounds, lr=0.05, iters=200)
    print("[Adam-FD] f_best:", res_adam.f_best, "x*:", res_adam.x_best)


if __name__ == "__main__":
    # Toggle which entrypoint you want to run:
    #main()          # CMA-ES + Adam-FD (gradient-free)
    main_autograd()   # Gradient-based with PyTorch autograd
