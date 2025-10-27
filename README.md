# sngp-sandbox

Implementation in pytorch to look into the proposed SNGP algorithm from Liu et. al (https://arxiv.org/pdf/2006.10108)

```
# SNGP Sandbox

This repository contains an implementation of the Spectral Normalization Gaussian Process (SNGP) algorithm proposed by Liu et al. in their paper [Simple and Principled Uncertainty Estimation with Deterministic Deep Learning via Distance Awareness](https://arxiv.org/pdf/2006.10108). The SNGP algorithm is designed to improve uncertainty estimation in deep learning models by incorporating distance-awareness and spectral normalization.

## Features

- **PyTorch Implementation**: The SNGP algorithm is implemented using PyTorch, making it easy to integrate into existing PyTorch-based projects.
- **Uncertainty Estimation**: Demonstrates how SNGP can be used to improve uncertainty estimation in deterministic deep learning models.
- **Customizable**: The implementation is modular and can be adapted for various use cases and datasets.

## Requirements

To run this code, you will need the following dependencies:

- Python 3.8+
- PyTorch 1.10+
- NumPy
- Matplotlib
- Other dependencies listed in `requirements.txt`

Install the dependencies using:
```

## Usage: 
Clone the repository:

```
git clone https://github.com/your-username/sngp-sandbox.git 
cd sngp-sandbox
```

Train the model:

`python train_models.py`

Run Notebook in `notebooks`:

`sngp-toy-comparison.ipynb`

References
Liu, J., Lin, Z., Padhy, S., Tran, D., Bedrax-Weiss, T., & Lakshminarayanan, B. (2020). Simple and Principled Uncertainty Estimation with Deterministic Deep Learning via Distance Awareness.
License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
Special thanks to Liu et al. for their work on the SNGP algorithm and the open-source community for their contributions to PyTorch and related libraries.