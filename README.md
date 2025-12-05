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

-   **`src/baseline_wang_fujin/`**: Contains the original baseline training scripts.
    -   `main_XJTU.py`, `main_MIT.py`, `main_HUST.py`, `main_TJU.py`: Baseline training scripts.
-   **`src/models/pinn.py`**: Standard PINN implementation (Baseline).
-   **`src/models/baselines.py`**: Other baseline models (MLP, CNN).
-   **`src/dataloaders/`**: Data loading utilities for XJTU, MIT, HUST, and TJU datasets.

> **Credit**: The baseline PINN code and data loading logic are adapted from [Wang Fujin's PINN4SOH repository](https://github.com/wang-fujin/PINN4SOH).

## Project Structure

```
├── src/
│   ├── qk/                     # [My Contribution] Quantum Kernel & QPINN Models
│   ├── baseline_wang_fujin/    # [Baseline] Original training scripts by Wang Fujin
│   ├── models/                 # [Baseline] Standard PINN & Baselines
│   ├── dataloader/             # [Baseline] Data Loading Utilities
│   └── utils/                  # General Utilities
├── data/                       # Dataset directory (XJTU, MIT, etc.)
├── docs/                       # Documentation
└── README.md                   # This file
```

## Usage

### Running Quantum PINN (My Contribution)
To train the Quantum PINN model on the XJTU dataset:

```bash
python src/qk/main_xjtu.py --xjtu_batch 2C --epochs 50
```

To run on other datasets:
```bash
python src/qk/main_mit.py --epochs 50
python src/qk/main_hust.py --epochs 50
python src/qk/main_tju.py --epochs 50
```

### Running Baseline PINN (Wang Fujin's Code)
To run the baseline PINN model using the scripts in `src/baseline_wang_fujin/`:

```bash
python src/baseline_wang_fujin/main_XJTU.py --batch 2C
```

## Requirements
-   Python 3.7+
-   PyTorch 1.7.1+
-   PennyLane (for Quantum simulations)
-   scikit-learn
-   numpy
-   pandas
-   matplotlib
