# 🧰 Adversarial Defense Experiment Toolkit

This repository reorganises the earlier collection of standalone evaluation scripts into a
modular Python package named `advdef`. It provides a reproducible experiment pipeline
for benchmarking adversarial attacks, defenses, and classifiers on ImageNet-like data.

## 🔧 Features

- Modular components for datasets, attacks, defenses, inference, and evaluation.
- YAML-based experiment configuration with Pydantic validation.
- Command-line interface for running experiments and queues.

### ⚔️ Available Attacks

- AutoAttack Suite

### 🛡️ Available Defenses

- SMoE (Steered Mixture-of-Experts)
- JPEG Compression
- Grayscale Conversion

## 🖥️ Installation

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

## 🧪 Running an Experiment

Configure an experiment by editing a YAML file (see `configs/resnet50_autoattack.yaml`).
By default the ImageNet dataset builder downloads the validation split into
`datasets/imagenet/val/` and the devkit (ground-truth labels) into
`datasets/imagenet/ILSVRC2012_devkit_t12/` the first time they are needed,
caching the archives under `downloads/`.
Override paths in the config only if you store the data elsewhere. Run the
experiment with:

```bash
advdef run configs/resnet50_autoattack.yaml
```

When you already have the validation set/devkit stored elsewhere, pass
`--imagenet-root /path/to/imagenet` to reuse it without editing configs.

Artifacts, metrics, configuration snapshots, and logs are stored in
`runs/<timestamp>_<slug>/`.

### ⏳ Queueing Multiple Experiments

Queue files describe a list of experiments (with optional overrides). Execute:

```bash
advdef queue configs/queues/sample_queue.yaml
```

Each job resolves its referenced config, applies overrides, and runs sequentially
by default.

## 🙏 Acknowledgements

This project uses Yi-Hsin Li's R-SMoE implementation for the SMoE defense component
(https://github.com/yihsinli/r-smoe), included via git submodule and distributed
under its original Gaussian-Splatting License (`external/r-smoe/LICENSE.md`).
