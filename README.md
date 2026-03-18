# CARN: Complexity-Aware Routing Network for Efficient and Adaptive Inference (PyTorch)

Official PyTorch implementation of **“CARN: Complexity-Aware Routing Network for Efficient and Adaptive Inference”**.

**Authors**

- **Rebati Gaire**  
  University of Illinois Chicago, Chicago, Illinois  
  `rrgaire@uic.edu`
- **Arman Roohi**  
  University of Illinois Chicago, Chicago, Illinois

---

## Abstract

Deep neural networks (DNNs) have achieved remarkable success across various domains, yet their rigid, static computation graphs lead to significant inefficiencies in realworld deployment. Standard architectures allocate equal computational resources to all inputs, disregarding their inherent complexity, which results in unnecessary computation for simple samples and suboptimal processing for complex ones. To address this, we propose the Complexity-Aware Routing Network (CARN), a novel framework that dynamically adjusts computational pathways based on input complexity. CARN integrates a self-supervised complexity estimation module that quantifies input difficulty using confidence, entropy, and computational cost, guiding a neural network-based routing mechanism to optimally assign task modules. The model is trained using a routing loss function that balances assignment accuracy and computational efficiency, mitigating expert starvation while preserving specialization. Extensive experiments on CIFAR10, CIFAR-100, and Tiny-ImageNet demonstrate that CARN achieves up to 4× reduction in computational cost and over 10× reduction in parameter movement while maintaining high accuracy compared to state-of-the-art static models.

---

## Repository layout

- `cifar10/`, `cifar100/`, `tinyimagenet/`: experiment code by dataset/backbone
- `test/`: minimal runnable example scripts
- `assets/`: images/tables for this README (place results here and reference as `assets/<file>`).

---

## Prerequisites

- Python 3.9+ (recommended)
- PyTorch + torchvision
- CUDA (optional, recommended for training)

> Note: Some scripts use additional packages (e.g., `wandb`, `thop`, `fvcore`, `tqdm`). See “Installation”.

---

## Installation

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision
pip install wandb thop fvcore tqdm numpy
```

If you prefer CUDA-specific wheels, install PyTorch using the official instructions for your platform.

---

## Training

This codebase contains multiple experiment entrypoints under each dataset/backbone directory (e.g., `cifar10/vgg19/sampler/main.py`).

Typical workflow:

1. Train (or provide) the feature extractor + task modules (experts).
2. Train the routing (sampler) network using the provided scripts.

Example (CIFAR-10 / VGG-19 sampler training):

```bash
python cifar10/vgg19/sampler/main.py \
  --data_path path/to/data \
  --checkpoint_dir path/to/checkpoints \
  --log_dir path/to/logs \
  --fe_ckpt path/to/feature_extractor.pth \
  --classifier1_ckpt path/to/classifier1.pth \
  --classifier2_ckpt path/to/classifier2.pth \
  --classifier3_ckpt path/to/classifier3.pth
```

---

## Inference

After training, you can evaluate the learned routing + task model using the corresponding `test.py` scripts (located under the relevant `*/sampler/` directories).

> The evaluation scripts currently expect checkpoint paths to be passed explicitly (no machine-specific absolute paths).

---

## Results

Add plots/tables to `assets/` and reference them here, for example:

- `assets/results_cifar10.png`
- `assets/results_tinyimagenet.png`

---

## Citation

If you use this codebase in your research, please cite:

```bibtex
@inproceedings{gaire2026carn,
  title     = {CARN: Complexity-Aware Routing Network for Efficient and Adaptive Inference},
  author    = {Gaire, Rebati and Roohi, Arman},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2026}
}
```

> Update the venue/year/metadata above to match the final published version of the paper.

