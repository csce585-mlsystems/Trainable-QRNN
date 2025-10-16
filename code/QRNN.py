import torch
import torch.nn as nn
import numpy as np
from qiskit_aer import AerSimulator
from qiskit import transpile

from SimpleBlock import SimpleBlock
from quantum_layerv3 import QuantumLayer


class QRNN(nn.Module):
    def __init__(self, n_qubits, repeat_blocks, in_dim, out_dim,
                 context_length, sequence_length, batch_size,
                 shots=4096, seed=0, grad_method="finite-diff",epsilon=1e-1,spsa_samples=2):
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
        self.spsa_samples = spsa_samples
        self.eps = epsilon

        assert n_qubits % 2 == 0, "n_qubits must be even"
        self.n_readout = n_qubits // 2

        self.qc, self.param_vectors, self.register_names = SimpleBlock(
            self.n_qubits,
            self.context_length,
            self.repeat_blocks,
            self.sequence_length,
            seed=self.seed
        )

        self.params_c = len(self.param_vectors[0])

        self.input_layer = nn.Linear(in_dim * context_length, self.params_c)
        CONSTANT = (context_length*(repeat_blocks)*np.ceil(in_dim))
        scale = np.pi / CONSTANT  # ~0.39 rad
        nn.init.uniform_(self.input_layer.weight, -scale, scale)
        nn.init.uniform_(self.input_layer.bias, -np.pi/(2*repeat_blocks), scale/(2*repeat_blocks))

        self.output_layer = nn.Linear(2**self.n_readout, out_dim)
        fan_in = 2 ** self.n_readout
        limit = np.sqrt(6 / fan_in) * 2.0   # smaller gain
        nn.init.uniform_(self.output_layer.weight, -limit, limit)
        nn.init.constant_(self.output_layer.bias, 0.5)

        self.sim = AerSimulator()
        self.compiled = transpile(self.qc, self.sim)

        self._backup_params = {}

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
        batch_size = x.shape[0]
        all_batch_outputs = []
        for b in range(batch_size):
            mapping = {}
            for t in range(self.sequence_length):
                pv = self.param_vectors[t]
                vals = x[b, t].detach().cpu().numpy().ravel()
                for i, p in enumerate(pv):
                    mapping[p] = float(vals[i])
            bound_circ = self.compiled.assign_parameters(mapping)
            job = self.sim.run(bound_circ, shots=int(self.shots))
            result = job.result()
            counts = result.get_counts()
            num_outcomes = 2 ** self.n_readout
            time_steps = self.sequence_length
            probs = np.zeros((time_steps, num_outcomes))
            for bitstring, c in counts.items():
                registers = bitstring.split()
                registers = registers[::-1]
                for t, reg_bits in enumerate(registers):
                    idx = int(reg_bits, 2)
                    probs[t, idx] += c / float(self.shots)
            all_batch_outputs.append(probs)
        return torch.tensor(np.array(all_batch_outputs), dtype=torch.float32)


    def forward(self, x):
        batch, time, _ = x.shape
        x_flat = x.reshape(batch * time, -1)
        encoded = self.input_layer(x_flat)
        encoded = encoded.view(batch, time, -1)
        out = QuantumLayer.apply(self, encoded, self.grad_method,self.eps,self.spsa_samples)
        return self.output_layer(out), out

    def _get_param(self, fullname):
        parts = fullname.split('.')
        obj = self
        for p in parts[:-1]:
            obj = getattr(obj, p)
        final = parts[-1]
        return getattr(obj, final)

    def _set_param(self, fullname, new_tensor):
        parts = fullname.split('.')
        obj = self
        for p in parts[:-1]:
            obj = getattr(obj, p)
        final = parts[-1]
        param_obj = getattr(obj, final)
        if isinstance(param_obj, torch.nn.Parameter):
            param_obj.data = new_tensor.to(param_obj.data.dtype).to(param_obj.data.device).clone()
        else:
            setattr(obj, final, new_tensor.clone().to(getattr(obj, final).dtype))
