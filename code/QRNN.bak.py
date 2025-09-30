import torch
import torch.nn as nn
import numpy as np
from qiskit_aer import AerSimulator
from qiskit import transpile

from ReuploadingBlock import ReuploadingBlock


class QRNN(nn.Module):
    def __init__(self, n_qubits, repeat_blocks, in_dim, out_dim,
                 context_length, sequence_length, batch_size,
                 shots=4096, seed=0):
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

        assert n_qubits % 2 == 0, "n_qubits must be even"

        self.n_readout = n_qubits // 2

        # --- weight initialization ---
        # input encoding weights: one (context_length x 3) matrix per qubit
        self.W_in = nn.Parameter(
            torch.empty(n_qubits, in_dim * context_length, 3).uniform_(-np.pi, np.pi)
        )

        # Each qubit has biases for (phi, theta, omega)
        self.W_bias = nn.Parameter(
            torch.empty(n_qubits, 3).uniform_(-np.pi, np.pi)
        )

        # Hidden weights: (n_qubits//2, 2) for CRY and CRX
        self.W_hidden = nn.Parameter(
            torch.empty(n_qubits // 2, 2).uniform_(-np.pi, np.pi)
        )

        # Entangling weights: (n_qubits//2,) for CRZ
        self.W_entangle = nn.Parameter(
            torch.empty(n_qubits // 2).uniform_(-np.pi, np.pi)
        )


        # output layer
        self.output_layer = nn.Linear(2**self.n_readout, out_dim)

        # simulator
        self.sim = AerSimulator()

    def forward(self, x):
        """
        x shape: (batch_size, sequence_length, in_dim)
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        all_batch_outputs = []

        for b in range(batch_size):
            # build circuit for this sample
            qc, reg_names = ReuploadingBlock(
                x[b].detach().cpu().numpy(),
                self.n_qubits,
                self.context_length,
                self.repeat_blocks,
                self.W_in.detach().cpu().numpy(),
                self.W_bias.detach().cpu().numpy(),
                self.W_hidden.detach().cpu().numpy(),
                self.W_entangle.detach().cpu().numpy(),
            )

            compiled = transpile(qc, self.sim)
            result = self.sim.run(compiled, shots=self.shots).result()
            counts = result.get_counts()

            # decode probabilities per timestep
            probs = np.zeros((seq_len, 2**self.n_readout))
            for bitstring, c in counts.items():
                bits = bitstring.replace(" ", "")[::-1]
                # split into timestep chunks
                for t, reg in enumerate(reg_names):
                    slice_bits = bits[t*self.n_readout:(t+1)*self.n_readout]
                    idx = int(slice_bits, 2)
                    probs[t, idx] += c / self.shots

            all_batch_outputs.append(probs)

        all_batch_outputs = torch.tensor(np.array(all_batch_outputs), dtype=torch.float32)

        # apply output layer timestep-wise
        final_out = self.output_layer(all_batch_outputs)
        return final_out[0]
