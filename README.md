# Adversarial Defense Experiment Toolkit

This repository reorganises the earlier collection of standalone evaluation scripts into a
modular Python package named `advdef`. It provides a reproducible experiment pipeline
for benchmarking adversarial attacks, defenses, and classifiers on ImageNet-like data.

## Layout

- `pyproject.toml` – package metadata and dependencies.
- `lib/advdef/` – Python package sources.
  - `config/` – Pydantic models describing experiment, dataset, attack, defense, model, inference and evaluation settings.
  - `core/` – pipeline execution, registries, run context, CLI helpers.
  - `datasets/` – dataset preparation utilities (e.g. ImageNet sampling).
  - `attacks/` – adversarial attack implementations (AutoAttack).
  - `defenses/` – first-party defenses (e.g. JPEG recompression, R-SMOE pipeline wrapper).
  - `inference/` – inference backends (timm models).
  - `evaluation/` – metric computation (ImageNet accuracies).
  - `utils/` – filesystem, serialisation, time, and environment helpers.
- `external/r-smoe/` – git submodule containing the vendor R-SMOE implementation consumed by the defense.
- `configs/` – example experiment YAML files.
  - `queues/` – queue definitions for batching multiple experiments.
- `old scripts/` – original one-off scripts preserved for reference.

## Installation

Create and activate a Python 3.10+ environment, then install the toolkit in editable mode:

```bash
pip install -e .
```

Ensure that additional dependencies (CUDA-enabled PyTorch, pyautoattack, the
`plyfile` package, and the R-SMOE submodule requirements) are installed in the
environment.

To use the R-SMOE defense, run the helper command once to build the CUDA
extensions and install the required Python packages in your current environment:

```bash
advdef setup r-smoe
```

If you prefer, you can follow the upstream instructions instead and create the
`r-smoe` Conda environment provided in `external/r-smoe/my_environment.yml`. The
command above captures the minimal additional packages required when reusing the
environment created for this toolkit.

## Running an Experiment

Configure an experiment by editing a YAML file (see `configs/resnet50_autoattack.yaml`).
By default the ImageNet dataset builder downloads the validation split into
`datasets/imagenet/val/` and the devkit into `datasets/imagenet/ILSVRC2012_devkit_t12/`
the first time they are needed, caching the archives under `downloads/`.
Override paths in the config only if you store the data elsewhere. Run the
experiment with:

```bash
advdef run configs/resnet50_autoattack.yaml
```

When you already have the validation set/devkit stored elsewhere, pass
`--imagenet-root /path/to/imagenet` to reuse it without editing configs.

Artifacts, metrics, configuration snapshots, and logs are stored in
`runs/<timestamp>_<slug>/`.

## Queueing Multiple Experiments

Queue files describe a list of experiments (with optional overrides). Execute:

```bash
advdef queue configs/queues/sample_queue.yaml
```

Each job resolves its referenced config, applies overrides, and runs sequentially
by default.

## Extending

- **Datasets / Attacks / Defenses / Inference / Evaluation**: implement a class
  that inherits from the corresponding base in `advdef.core.pipeline`, decorate it with
  `@register_*`, and expose the module in its package `__init__.py`.
- **Configuration**: define new parameters in the relevant Pydantic models under `advdef.config`.
- **CLI**: additional commands can be added to `advdef.cli`.

Refer to the existing implementations (AutoAttack dataset + attack, R-SMOE defense,
Timm inference, ImageNet evaluation) for concrete patterns. The registries allow
dynamically referencing components via their `type` string in experiment configs.
