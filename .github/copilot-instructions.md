# Copilot / AI Agent Instructions for OpenSTL

Quick, actionable guidance to help an AI code agent become productive in this repo.

## Big picture (what matters)
- Architecture layers:
  - openstl/core: training plugins, schedulers, metrics.
  - openstl/datasets: dataset loaders and data pipelines.
  - openstl/modules & openstl/models: reusable building blocks and network definitions.
  - openstl/methods: high-level training/prediction strategy wrappers (use `method_maps` to find mapping).
  - openstl/api (e.g., `exp.py` -> `BaseExperiment`): experiment lifecycle (data, model, Trainer setup).
- Experiments use PyTorch Lightning (`lightning.Trainer`) and save results under `work_dirs/<ex_name>/` with checkpoints at `work_dirs/<ex_name>/checkpoints/`.

## How to run (concrete commands)
- Setup environment (recommended):
  - conda env create -f environment.yml
  - conda activate OpenSTL
  - python setup.py develop
- Prepare data: scripts in `tools/prepare_data/` (e.g., `bash tools/prepare_data/download_mmnist.sh`).
- Train (example):
  - python tools/train.py -d mmnist -c configs/mmnist/simvp/SimVP_gSTA.py --ex_name mmnist_simvp_gsta
  - The code infers config path with `-d <dataname>` and `-m <method>` when `-c` is omitted.
- Test / load checkpoint (examples):
  - python tools/test.py -c configs/mmnist/simvp/SimVP_gSTA.py --test
  - python tools/test.py -c configs/mmnist/simvp/SimVP_gSTA.py --test --ckpt_path work_dirs/<ex_name>/checkpoints/best.ckpt
  - Note: `test.py` contains logic to robustly load ckpt files (strips `module.` and `model.` prefixes and uses a torch.load option compatible with newer PyTorch versions).
- Run unit tests: pytest -q (tests live under `tests/`).

## Configs & CLI behavior (where to change hyperparams)
- Config files are Python files under `configs/<dataname>/...` (e.g., `configs/mmnist/simvp/SimVP_gSTA.py`). They define global variables (lr, batch_size, model_type, etc.).
- CLI parser: `openstl/utils/parser.py` provides arguments and defaults. `tools/train.py` merges CLI args with config file using `update_config`.
- Use `--overwrite` to let CLI args override config file values fully; otherwise only selected keys are updated.

## Checkpoints & debugging tips
- Default best checkpoint path: `work_dirs/<ex_name>/checkpoints/best.ckpt`.
- Resume training: `python tools/train.py ... --ckpt_path path/to/ckpt` (train.py passes `ckpt_path` through to Trainer.fit).
- When loading external ckpts, mismatch errors are common — use `--ckpt_path` to trigger `test.py`'s prefix-cleanup loader or manually inspect state dict keys and remove DDP/lightning prefixes.

## Project-specific conventions & patterns
- Method -> Model -> Modules decomposition: implement a new algorithm by adding a `method` (high-level training) in `openstl/methods`, networks in `openstl/models`, and reusable layers in `openstl/modules`.
- Dataset integration: add a loader under `openstl/datasets` and a `tools/prepare_data` script for large downloads/processing.
- Naming: `--method` is case-insensitive but normalized to lowercase in `BaseExperiment`.
- GPUs: CLI `--gpus` expects a list of integers (e.g., `--gpus 0 1`); Lightning `Trainer` is configured in `BaseExperiment._init_trainer` (strategy `'auto'` by default).
- Performance diagnostics: use `--fps` to measure throughput in `BaseExperiment.display_method_info` and FLOPs are computed with `fvcore` when requested.

## Tests and CI notes
- Unit tests: `pytest` runs model- and util-focused tests under `tests/`.
- Keep changes backward-compatible with current config variables (configs are imported as plain modules and expected globals).

## Common PR/bug-fix patterns (concrete examples to follow)
- If a breakpoint or mismatch occurs while loading ckpt: inspect saved `state_dict` keys, remove `module.` or `model.` prefixes and reload (the repo already includes robust loader logic in `tools/test.py`).
- Changes to model shapes often need corresponding config updates in `configs/*` and unit tests added under `tests/test_models/`.
- For distributed training bugs, check `--gpus`, `--dist` semantics and Lightning strategy; verify DDP-induced name prefixes and batch sharding.

## Files to inspect first when asked to modify behavior
- `tools/train.py`, `tools/test.py` (entrypoints)
- `openstl/api/exp.py` (BaseExperiment lifecycle)
- `openstl/utils/parser.py` and `openstl/utils/config_utils.py` (CLI & config semantics)
- `configs/` (dataset/method defaults)
- `openstl/datasets/`, `openstl/methods/`, `openstl/models/` (to understand data flow and model composition)

---
If anything here is unclear or you'd like examples expanded (e.g., a checklist for adding a new dataset or a sample PR description), tell me which part to expand and I'll iterate.  

(Notes: keep this file concise — it targets AI agents and reviewers, not end users.)