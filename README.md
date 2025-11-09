# 🧰 Adversarial Defense Experiment Toolkit

The Adversarial Defense Experiment Toolkit is a modular framework for testing different input transformation defenses against adversarial attacks on image classification models. It supports configuring experiments via YAML files, running them through a command-line interface, and logging results for analysis.

## 🔧 Features

- Modular components for datasets, attacks, defenses, inference, and evaluation.
- YAML-based experiment configuration with Pydantic validation.
- Command-line interface for running experiments and queues.

### ⚔️ Available Attacks

- AutoAttack Suite (APGD-CE, APGD-T, FAB, Square)
- PGD (ℓ∞/ℓ₂)
- FGSM
- Carlini & Wagner (ℓ₂)
- DeepFool

### 🛡️ Available Defenses

- SMoE (Steered Mixture-of-Experts)
- BM3D Denoising (CPU or CUDA backend)
- JPEG Compression
- Grayscale Conversion
- Crop & Resize Transformation
- Horizontal Flip
- Bit-depth Reduction
- Total Variation Minimization (TVM)
- Low-pass Filtering

## 🖥️ Installation

Create and activate a Python 3.10+ environment, then install the toolkit in editable mode:

```bash
pip install -e .
```

Ensure that additional dependencies (CUDA-enabled PyTorch, pyautoattack, the
`plyfile` package, and the R-SMOE submodule requirements) are installed in the
environment.

To install optional attack backends (PGD, FGSM, CW, DeepFool via `torchattacks`), run:

```bash
pip install -e '.[attacks]'
```

To use the R-SMOE defense, run the helper command once to build the CUDA
extensions and install the required Python packages in your current environment:

```bash
advdef setup r-smoe
```

If you prefer, you can follow the upstream instructions instead and create the
`r-smoe` Conda environment provided in `external/r-smoe/my_environment.yml`. The
command above captures the minimal additional packages required when reusing the
environment created for this toolkit.

To enable the BM3D defense install at least one backend:

- CPU reference implementation:

  ```bash
  pip install -e '.[denoisers]'
  ```

- CUDA backend (`external/bm3d-gpu/` submodule using the upstream CLI project):

  ```bash
  git submodule update --init --recursive external/bm3d-gpu
  advdef setup bm3d-gpu
  ```

  The helper configures and builds the
  [`bm3d-gpu`](https://github.com/DawyD/bm3d-gpu) CLI binary (requires the CUDA
  toolkit plus CMake). After building, follow the defense config in
  `configs/resnet50_autoattack_bm3d.yaml` (set `backend: cli`, `cli_binary`,
  `cli_color_mode`, etc.) and adjust parameters as needed for your environment.

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
# increase verbosity if needed
advdef run configs/resnet50_autoattack.yaml --log-level info
```

Resume a partially completed run by reusing its `--run-name` and adding `--resume`. Cached artifacts
allow the pipeline to skip completed stages:

```bash
advdef run configs/resnet50_autoattack.yaml --run-name 20240122_resnet50_autoattack --resume
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
by default. Provide `--resume` when re-running a queue that specifies `run_name`
values to continue unfinished jobs without repeating completed ones.

## 🙏 Acknowledgements

This project uses Yi-Hsin Li's R-SMoE implementation for the SMoE defense component
(https://github.com/yihsinli/r-smoe), included via git submodule and distributed
under its original Gaussian-Splatting License (`external/r-smoe/LICENSE.md`).
