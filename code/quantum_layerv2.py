import torch
from torch.autograd import Function
import numpy as np
from qiskit_aer.primitives import Sampler as AerSampler
from qiskit_algorithms.gradients import FiniteDiffSamplerGradient, SPSASamplerGradient


class QuantumLayer(Function):
    @staticmethod
    def forward(ctx, qrnn_module, x, grad_method="finite-diff", epsilon=1e-1, spsa_samples=3):
        ctx.qrnn = qrnn_module
        ctx.grad_method = grad_method
        ctx.epsilon = epsilon
        ctx.spsa_samples = spsa_samples
        ctx.save_for_backward(x)

        # Use the QRNN's own _quantum_forward which uses counts and your proven bitstring parsing
        with torch.no_grad():
            out = qrnn_module._quantum_forward(x)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        qrnn = ctx.qrnn
        grad_method = ctx.grad_method
        eps = ctx.epsilon
        spsa_samples = ctx.spsa_samples
        (x_saved,) = ctx.saved_tensors

        batch_size = x_saved.shape[0]
        time_steps = qrnn.sequence_length
        params_per_timestep = len(qrnn.param_vectors[0])
        total_params = params_per_timestep * time_steps
        n_readout = qrnn.n_readout
        num_outcomes_per_timestep = 2 ** n_readout
        total_bits = n_readout * time_steps

        # gradient w.r.t. the encoded activations (same shape as x_saved)
        grad_x = torch.zeros_like(x_saved)

        sampler = AerSampler(run_options={"shots": qrnn.shots})
        if grad_method.lower() in ("finite-diff", "finite_diff", "finite"):
            grad_calc = FiniteDiffSamplerGradient(sampler, epsilon=eps)
        else:
            grad_calc = SPSASamplerGradient(sampler, epsilon=eps, batch_size=spsa_samples)

        for b in range(batch_size):
            # Build parameter list in same ordering as _quantum_forward
            param_values = []
            for t in range(time_steps):
                pv = qrnn.param_vectors[t]
                vals = x_saved[b, t].detach().cpu().numpy().ravel()
                for i, _ in enumerate(pv):
                    param_values.append(float(vals[i]))

            # run sampler-gradient primitive to get derivatives d(full_outcome_prob)/dparam_j
            res_job = grad_calc.run(circuits=[qrnn.compiled], parameter_values=[param_values])
            try:
                res = res_job.result()
            except Exception:
                res = res_job

            # expect res.gradients[0] to be a sequence indexed by parameter,
            # each entry either a dict {full_outcome_index: dprob/dparam} or array-like
            if hasattr(res, "gradients"):
                gradients_list = res.gradients[0]
            elif hasattr(res, "gradient"):
                gradients_list = res.gradient[0]
            else:
                gradients_list = res  # best-effort fallback

            # accumulate d(prob_t, idx)/dparam_j
            grads_per_param = np.zeros((total_params, time_steps, num_outcomes_per_timestep), dtype=float)

            for j in range(total_params):
                try:
                    grad_entry = gradients_list[j]
                except Exception:
                    grad_entry = {}

                # grad_entry can be dict mapping full outcome index -> derivative
                if isinstance(grad_entry, dict):
                    for full_idx, grad_val in grad_entry.items():
                        bits = format(int(full_idx), 'b').zfill(total_bits)
                        registers = [bits[i * n_readout:(i + 1) * n_readout] for i in range(time_steps)]
                        # match your forward ordering: reverse the register list
                        registers = registers[::-1]
                        for t, reg_bits in enumerate(registers):
                            idx = int(reg_bits, 2)
                            grads_per_param[j, t, idx] += float(grad_val)
                else:
                    # array-like: index = full outcome integer
                    try:
                        arr = np.asarray(grad_entry)
                        for full_idx, grad_val in enumerate(arr):
                            bits = format(int(full_idx), 'b').zfill(total_bits)
                            registers = [bits[i * n_readout:(i + 1) * n_readout] for i in range(time_steps)]
                            registers = registers[::-1]
                            for t, reg_bits in enumerate(registers):
                                idx = int(reg_bits, 2)
                                grads_per_param[j, t, idx] += float(grad_val)
                    except Exception:
                        pass

            # Now compute dL/dparam_j = sum_{t,idx} dL/dprob_{t,idx} * dprob_{t,idx}/dparam_j
            gout = grad_output[b].detach().cpu().numpy()  # shape (time_steps, num_outcomes_per_timestep)
            dL_dparam = np.zeros((total_params,), dtype=float)
            for j in range(total_params):
                dL_dparam[j] = np.sum(gout * grads_per_param[j])

            # reshape to (time_steps, params_per_timestep) and return this as gradient w.r.t the encoded activations
            grad_encoded = dL_dparam.reshape((time_steps, params_per_timestep))

            device = x_saved.device
            dtype = x_saved.dtype
            grad_encoded_torch = torch.tensor(grad_encoded, dtype=dtype, device=device)

            grad_x[b] = grad_encoded_torch

        # Return tuple matching forward inputs: (qrnn_module, x, grad_method, epsilon, spsa_samples)
        return (None, grad_x, None, None, None)
