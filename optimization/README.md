
# optimization/
Minimal, modular optimization wrapper for **causaliT + proT**.

## Files
- `predictors.py` — adapters:
  - `SCMPredictor`: wrap a known ground-truth function `P -> Y` (great for validation)
  - `ModelPredictor`: wrap a trained proT model via two small callables (`build_input`, `run_model`)

- `objectives.py` — define your optimization goal (maximize Y, or hit a target Y\_target).

- `optimizers.py` — two ready-to-use optimizers:
  - `cma_es` (uses `cma` if available, else a simple NES fallback)
  - `adam` (finite-difference gradients; OK for small dimension)

- `run_example_scm.py` — **works out of the box**. Optimizes a known analytic SCM (no model needed).

- `run_with_model_template.py` — template to connect to a **trained proT** checkpoint. 
  You only need to implement `load_model` and possibly tweak `build_input` for your dataset.

## Quickstart: validate pipeline with known SCM
```bash
cd optimization
python run_example_scm.py
```
You should see both CMA-ES and Adam-FD converging close to the analytic optimum.

## Use with your trained proT model
1. Train the transformer on a synthetic dataset (e.g., `data/example/`).
2. Edit `run_with_model_template.py`:
   - `DATA_DIR` → your dataset folder
   - `CONTROLLABLE` + bounds → the P variables you want to optimize
   - `CKPT_PATH` → path to your Lightning/PyTorch checkpoint
   - Implement `load_model(ckpt_path)` and, if needed, adjust `run_model(...)`

3. Run:
```bash
python run_with_model_template.py
```

This keeps the optimization layer **cleanly separated** (as Francesco requested). You can import utilities from `proT` without mixing code.
