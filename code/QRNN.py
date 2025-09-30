import torch
import torch.nn as nn
import numpy as np
from qiskit_aer import AerSimulator
from qiskit import transpile

from ReuploadingBlock import ReuploadingBlock
from quantum_layer import QuantumLayer


class QRNN(nn.Module):
    def __init__(self, n_qubits, repeat_blocks, in_dim, out_dim,
                 context_length, sequence_length, batch_size,
                 shots=4096, seed=0, grad_method="finite-diff"):
        super(QRNN, self).__init__()

        self.n_qubits = n_qubits
        self.repeat_blocks = repeat_blocks
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.context_length = context_length
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.shots = shots
        self.seed = seed
        self.grad_method = grad_method

        assert n_qubits % 2 == 0, "n_qubits must be even"
        self.n_readout = n_qubits // 2

        # --- weights ---
        self.W_in = nn.Parameter(
            torch.empty(n_qubits, context_length, 3).uniform_(-np.pi, np.pi)
        )
        self.W_bias = nn.Parameter(torch.empty(n_qubits, 3).uniform_(-np.pi, np.pi))
        self.W_hidden = nn.Parameter(torch.empty(n_qubits // 2, 2).uniform_(-np.pi, np.pi))
        self.W_entangle = nn.Parameter(torch.empty(n_qubits // 2).uniform_(-np.pi, np.pi))

        # output layer
        self.output_layer = nn.Linear(2**self.n_readout, out_dim)

        # simulator
        self.sim = AerSimulator()

        # backup storage for gradient calculations
        self._backup_params = {}

    # ------------------------
    # Internal helpers
    # ------------------------
    def _backup(self):
        self._backup_params = {name: param.clone().detach()
                               for name, param in self.named_parameters()}

    def _restore_params(self):
        for name, param in self.named_parameters():
            if name in self._backup_params:
                param.data = self._backup_params[name].data.clone()

    def _set_param(self, name, new_value):
        getattr(self, name).data = new_value.clone()

    def _quantum_forward(self, x):
        """Run a forward pass through Aer (no autograd)."""
        batch_size = x.shape[0]
        all_batch_outputs = []

        for b in range(batch_size):
            qc, reg_names = ReuploadingBlock(
                x[b].detach().cpu().numpy(),
                self.n_qubits,
                self.context_length,
                self.repeat_blocks,
                self.W_in.detach().cpu().numpy(),
                self.W_bias.detach().cpu().numpy(),
                self.W_hidden.detach().cpu().numpy(),
                self.W_entangle.detach().cpu().numpy(),
                seed=self.seed
            )

            compiled = transpile(qc, self.sim)
            result = self.sim.run(compiled, shots=self.shots).result()
            counts = result.get_counts()

            num_outcomes = 2 ** self.n_readout
            time_steps = self.sequence_length
            probs = np.zeros((time_steps, num_outcomes))

            for bitstring, c in counts.items():
                # Split bitstring into registers ("01 11 00" → ["01","11","00"])
                registers = bitstring.split()
                registers = registers[::-1]  # Qiskit leftmost = last register

                for t, reg_bits in enumerate(registers):
                    idx = int(reg_bits, 2)
                    probs[t, idx] += c / self.shots


            all_batch_outputs.append(probs)

        return torch.tensor(np.array(all_batch_outputs), dtype=torch.float32)


    def forward(self, x):
        out = QuantumLayer.apply(self, x, self.grad_method)
        return self.output_layer(out)
