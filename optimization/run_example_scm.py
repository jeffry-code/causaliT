
"""
Example: optimize over P on the known SCM (ground truth) to validate your pipeline.
This does not require a trained model; useful to check optimizers and cost definitions.
"""
from __future__ import annotations
import numpy as np
from predictors import SCMPredictor, Bounds
from objectives import Objective
from optimizers import cma_es, adam

# --- Define a simple ground-truth SCM mapping P -> Y (you can swap with your scm.SCM) ---
# Example: Y = -(P1-1)^2 - (P2+2)^2 + 5  (maximum at [1, -2])
from scm_ds.datasets import ds_scm_1_to_1_ct  # uses the same SCM you generated

CONTROLLABLE = ["P1","P2","P3","P4","P5"]  # or a subset in your chosen order

def _zero_noise(scm):
    # zero noise for a single evaluation
    return {v: np.zeros(1, dtype=float) for v in scm.specs.keys()}

def ground_truth(P: np.ndarray) -> float:
    # Intervene on P, evaluate Y deterministically (eps=0)
    base = ds_scm_1_to_1_ct.scm
    scm_i = base.do({var: float(P[i]) for i, var in enumerate(CONTROLLABLE)})
    ctx = scm_i.forward(context={}, eps_draws=_zero_noise(scm_i))
    return float(ctx["Y"].reshape(-1)[0])

# Update bounds/init accordingly
lb = np.array([-3.0]*len(CONTROLLABLE))
ub = np.array([+3.0]*len(CONTROLLABLE))
x0 = np.zeros(len(CONTROLLABLE))
bounds = (lb, ub)

predictor = SCMPredictor(ground_truth)

# We want to MAXIMIZE Y, so Objective will MINIMIZE -Y by default (maximize=True).
obj = Objective(predictor, maximize=True)

# --- CMA-ES (gradient-free) ---
res_cma = cma_es(obj, x0=x0, sigma0=1.0, bounds=bounds, max_iters=200)
print("[CMA-ES] best f (negated objective):", res_cma.f_best, "x*:", res_cma.x_best)

# --- Adam with finite-difference gradients ---
res_adam = adam(obj, x0=x0, bounds=bounds, lr=0.05, iters=200)
print("[Adam-FD] best f:", res_adam.f_best, "x*:", res_adam.x_best)
