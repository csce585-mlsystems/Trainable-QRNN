# Introduction
Quantum Neural Networks (QNNs) remains a promising topic of interest in the field of Machine Learning (ML). With the advent of new simulation tools, it is now possible to run utility scale simulations of Quantum Circuits (QCs). In particular, the class of QCs known as Variational Quantum Circuits (VQCs) are of great interest to the Quantum Machine Learning (QML) community, and offer quantum analogue to the classical Neural Network (NN). With that said, the different types of hybrid classical-quantum models that can be feasibly constructed and trained remains largely unexplored. In addition, the difficulties associated with training these models on both classical computers and Quantum Processing Units (QPUs) warrants further investigation, as the cost currently associated with running on quantum hardware is very high.

In light of these unknowns, we propose a benchmark of the popular VQC training methods Simulataneous Perturbation Stochastic Approximation (SPSA) and parameter-shift within an end-to-end hybrid trainable Quantum Recurrent Neural Network (QRNN) model. Using data from the chaotic Kuramoto-Sivashinsky system (shown below), we will develop a time-series forecasting model and benchmark the different training methods. This study will shed light on the differing methods used to train QRNNs, and give an idea of which method results in lower loss and/or higher efficiency. By doing this analysis, we hope to shed light on what the best practices are for training QRNNs as QPUs become more accessible and more performant.

<img width="956" height="236" alt="Screenshot 2025-09-09 211555" src="https://github.com/user-attachments/assets/f078971d-a8e6-4389-8641-5e8afef08b62" />


# Related Works

Some work has been done on this in the past by the current authors. Specifically, the authors would like to implement the QRNN model described in Reference 1 as a trainable PyTorch layer. This QRNN model was tested and run on real quantum hardware, and demonstrates the necessary components for a recurrent neural network (RNN). The SPSA algorithm is described in Reference 2 and gives details on how to implement it for arbitrary optimization problems. Lastly, the parameter-shift method for QC gradient calculation is described in Reference 3 and describes this method in great detail.

# Hypothesis
Generally, the SPSA algorithm is assumed to be the best algorithm for VQC optimization, as it requires only $$O(n)$$ circuit evaluations to compute a gradient, where $$n$$ is the number of optimization steps. In contrast, the parameter-shift method requires $$O(pn)$$, where $$p$$ is the number of parameters in the VQC. As the number of parameters increases, the runtime of the parameter-shift method will increase, while the evaluations required by the SPSA algorithm will remain the same. This sort of runtime efficiency is very useful when the cost of both simulating and deploying VQCs onto quantum hardware remains high. However, it is known that training recurrent neural networks (RNNs) can be difficult (Reference 4), and it is not known whether these approximate methods such as SPSA hold up in a hybrid classical-quantum model with a recurrent quantum component. Knowing this, we believe there may be significant differences between the approximate and exact method, with the higher asymptotic runtime of the parameter-shift method possibly offering unforeseen improvements in optimization steps required and loss. This would run counter to conventional VQC training best practices and offer insight on how to train a QRNN.

# Goals


# References 
1. Connerty, E.L., Evans, E.N., Angelatos, G., Narayanan, V. (2025). Quantum Observers: A NISQ Hardware Demonstration of Chaotic State Prediction Using Quantum Echo-state Networks. arXiv preprint arXiv:2505.06799.
2. Spall, J.C. (1998). AN OVERVIEW OF THE SIMULTANEOUS PERTURBATION METHOD FOR EFFICIENT OPTIMIZATION. Johns Hopkins Apl Technical Digest, 19, 482-492.
3. Mitarai, K., Negoro, M., Kitagawa, M., & Fujii, K. (2018). Quantum circuit learning. Physical Review. A/Physical Review, A, 98(3). https://doi.org/10.1103/physreva.98.032309
4. Pascanu, R., Mikolov, T., Bengio, Y. (2012). On the difficulty of training Recurrent Neural Networks. arXiv preprint arXiv:1211.5063.
