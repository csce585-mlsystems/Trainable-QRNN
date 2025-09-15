# Project Title

## Benchmarking SPSA and Parameter-Shift Training Methods for Quantum Recurrent Neural Networks

## Group Info

* Erik Connerty

  * Email: [erikc@cec.sc.edu](mailto:erikc@cec.sc.edu)
* Michael Stewart

  * Email: [mcs46@email.sc.edu](mailto:mcs46@email.sc.edu)

## Project Summary/Abstract

Quantum Neural Networks (QNNs) are a promising area within Quantum Machine Learning (QML), particularly Variational Quantum Circuits (VQCs) which serve as quantum analogues to classical neural networks. In this project, we propose benchmarking two popular VQC training methods—Simultaneous Perturbation Stochastic Approximation (SPSA) and parameter-shift—within a hybrid Quantum Recurrent Neural Network (QRNN) model. Using chaotic time-series data from the Kuramoto-Sivashinsky system, we aim to evaluate their efficiency, loss performance, and applicability to hybrid quantum-classical models. This study will help establish best practices for training QRNNs as quantum hardware becomes more widely available.

## Problem Description

* Problem description: Training hybrid quantum-classical models, particularly QRNNs, remains challenging due to high computational cost and limited understanding of efficient optimization strategies. We benchmark SPSA and parameter-shift training methods to identify trade-offs in efficiency and accuracy.
* Motivation

  * Explore efficient training strategies for hybrid quantum models.
  * Assess scalability of approximate versus exact gradient methods.
  * Contribute to best practices for quantum machine learning as QPUs advance.
* Challenges

  * High cost of running quantum hardware and simulations.
  * Training instability in recurrent neural networks.
  * Balancing runtime efficiency with optimization quality.

## Contribution

### \[`Extension of existing work`], \[`Novel Contribution`]

We build upon prior work in QRNNs (Reference 1) and VQC optimization methods to benchmark SPSA (Reference 2) and parameter-shift (Reference 3) within a trainable PyTorch layer implementation. Specifically, we contribute:

* Development of a PyTorch-based QRNN layer for benchmarking.
* Integration and comparison of SPSA and parameter-shift optimization methods.
* Analysis of trade-offs in efficiency, accuracy, and applicability to hybrid models.

## References
1. Connerty, E.L., Evans, E.N., Angelatos, G., Narayanan, V. (2025). Quantum Observers: A NISQ Hardware Demonstration of Chaotic State Prediction Using Quantum Echo-state Networks. arXiv preprint arXiv:2505.06799.
2. Spall, J.C. (1998). AN OVERVIEW OF THE SIMULTANEOUS PERTURBATION METHOD FOR EFFICIENT OPTIMIZATION. Johns Hopkins Apl Technical Digest, 19, 482-492.
3. Mitarai, K., Negoro, M., Kitagawa, M., & Fujii, K. (2018). Quantum circuit learning. Physical Review. A/Physical Review, A, 98(3). https://doi.org/10.1103/physreva.98.032309
4. Pascanu, R., Mikolov, T., Bengio, Y. (2012). On the difficulty of training Recurrent Neural Networks. arXiv preprint arXiv:1211.5063.

# < The following is only applicable for the final project submission >  

## Dependencies  
### Include all dependencies required to run the project. Example:  
- Python 3.11  
- Ubuntu 22.04  

For Python users: Please use [uv](https://docs.astral.sh/uv/) as your package manager instead of `pip`. Your repo must include both the `uv.lock` and `pyproject.toml` files.  

## Directory Structure  
Example:  
```
|- data (mandatory)
|- src (mandatory)
|   |- model.py
|   |- example.py
|- train.py
|- run.py (mandatory)
|- result.py (mandatory)
```

⚠️ Notes:  
- All projects must include the `run.<ext>` script (extension depends on your programming language) at the project root directory. This is the script users will run to execute your project.  
- If your project computes/compares metrics such as accuracy, latency, or energy, you must include the `result.<ext>` script to plot the results.  
- Result files such as `.csv`, `.jpg`, or raw data must be saved in the `data` directory.  

## How to Run  
- Include all instructions (`commands`, `scripts`, etc.) needed to run your code.  
- Provide all other details a computer science student would need to reproduce your results.  

Example:  
- Download the [DATASET](dataset_link)
  ```bash
  wget <URL_of_file>
  ```

- To train the model, run:  
  ```bash
  python train.py
  ```  
- To plot the results, run:  
  ```bash
  python result.py
  ```  

## Demo  
- All projects must include video(s) demonstrating your project.  
- Please use annotations/explanations to clarify what is happening in the demo.  
---
