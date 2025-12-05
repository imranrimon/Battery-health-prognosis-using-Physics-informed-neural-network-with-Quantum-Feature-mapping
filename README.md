# Battery Health Prognosis using Physics-Informed Neural Network with Quantum Feature Mapping

This project implements a **Physics-Informed Neural Network (PINN)** for battery health prognosis, enhanced with **Quantum Feature Mapping**.

## Project Overview

The core innovation of this project is the integration of **Quantum Kernels** into the PINN architecture to improve feature extraction and prediction accuracy for battery State of Health (SOH).

### My Contributions (Quantum PINN)
The following directories and files contain the novel Quantum PINN implementation:

-   **`src/qk/`**: Contains the core Quantum Kernel implementation and QPINN model definitions.
    -   `model_qpin.py`: The Quantum Physics-Informed Neural Network (QPINN) model.
    -   `qkernel.py`: Implementation of Quantum Feature Maps and Kernels.
    -   `main_xjtu.py`, `main_mit.py`, `main_hust.py`, `main_tju.py`: Main training scripts for the respective datasets using the Quantum model.
    -   `main_xjtu_timed.py`: Performance benchmarking script for the Quantum model.

### Baseline / Reference Code
This project builds upon the baseline PINN implementation by **Wang Fujin**. The following components serve as the baseline and data infrastructure:

-   **`src/models/pinn.py`**: Standard PINN implementation (Baseline).
-   **`src/models/baselines.py`**: Other baseline models (MLP, CNN).
-   **`src/dataloaders/`**: Data loading utilities for XJTU, MIT, HUST, and TJU datasets.
-   **`src/spikan/main.py`**: Main training script for the baseline PINN.

> **Credit**: The baseline PINN code and data loading logic are adapted from [Wang Fujin's PINN4SOH repository](https://github.com/wang-fujin/PINN4SOH).

## Project Structure

```
├── src/
│   ├── qk/             # [My Contribution] Quantum Kernel & QPINN Models
│   ├── models/         # [Baseline] Standard PINN & Baselines
│   ├── dataloader/     # [Baseline] Data Loading Utilities
│   ├── spikan/         # [Baseline] Main script for standard PINN
│   └── utils/          # General Utilities
├── data/               # Dataset directory (XJTU, MIT, etc.)
├── docs/               # Documentation
└── README.md           # This file
```

## Usage

### Running Quantum PINN (My Contribution)
To train the Quantum PINN model on the XJTU dataset:

```bash
python src/qk/main_xjtu.py --xjtu_batch 2C --epochs 50
```

### Running Baseline PINN
To run the baseline PINN model:

```bash
python src/spikan/main.py --dataset XJTU --batch 2C
```

## Requirements
-   Python 3.7+
-   PyTorch 1.7.1+
-   PennyLane (for Quantum simulations)
-   scikit-learn
-   numpy
-   pandas
-   matplotlib
