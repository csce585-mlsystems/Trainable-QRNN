import torch
from torch.autograd import Function
import numpy as np

class QuantumLayer(Function):
    @staticmethod
    def forward(ctx, qrnn_module, x, grad_method="spsa", epsilon=1e-1, spsa_samples=3):
        ctx.qrnn = qrnn_module
        ctx.grad_method = grad_method
        ctx.epsilon = float(epsilon)
        ctx.spsa_samples = int(spsa_samples)
        ctx.save_for_backward(x)

        with torch.no_grad():
            out = qrnn_module._quantum_forward(x)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        qrnn = ctx.qrnn
        grad_method = ctx.grad_method.lower()
        eps = ctx.epsilon
        spsa_samples = ctx.spsa_samples
        (x_saved,) = ctx.saved_tensors

        batch_size = int(x_saved.shape[0])
        time_steps = int(qrnn.sequence_length)
        params_per_timestep = int(len(qrnn.param_vectors[0]))
        total_params = params_per_timestep * time_steps
        n_readout = int(qrnn.n_readout)
        num_outcomes_per_timestep = 2 ** n_readout
        total_bits = n_readout * time_steps

        grad_x = torch.zeros_like(x_saved)

        def counts_to_probs(counts_dict):
            """Parse counts dict into (time_steps, num_outcomes) probs using the same register splitting as forward."""
            num_outcomes = 2 ** n_readout
            probs = np.zeros((time_steps, num_outcomes), dtype=float)
            if not counts_dict:
                return probs
            for bitstring, c in counts_dict.items():
                registers = bitstring.split()
                registers = registers[::-1]
                for t, reg_bits in enumerate(registers):
                    idx = int(reg_bits, 2)
                    probs[t, idx] += c / float(qrnn.shots)
            return probs

        sim = qrnn.sim
        param_objs = list(qrnn.compiled.parameters)
        # ensure we only use as many parameter objects as total_params
        if len(param_objs) > total_params:
            param_objs = param_objs[:total_params]

        seed = int(getattr(qrnn, "seed", 0) or 0)
        rng = np.random.default_rng(seed)

        # --- SPSA (unchanged) ---
        if grad_method.startswith("spsa"):
            m = max(1, int(spsa_samples))
            c = float(eps)

            # flatten base param vectors per sample
            batch_param_values = []
            for b in range(batch_size):
                flat = []
                for t in range(time_steps):
                    pv = qrnn.param_vectors[t]
                    vals = x_saved[b, t].detach().cpu().numpy().ravel()
                    for i, _ in enumerate(pv):
                        flat.append(float(vals[i]))
                batch_param_values.append(np.array(flat, dtype=float))

            all_dL_dparam = np.zeros((batch_size, total_params), dtype=float)

            for b in range(batch_size):
                base = batch_param_values[b]
                g_est = np.zeros((total_params,), dtype=float)
                gout = grad_output[b].detach().cpu().numpy()

                for k in range(m):
                    delta = rng.choice([-1.0, 1.0], size=(total_params,))
                    p_plus = base + c * delta
                    p_minus = base - c * delta

                    mapping_plus = {param_objs[i]: float(p_plus[i]) for i in range(len(param_objs))}
                    mapping_minus = {param_objs[i]: float(p_minus[i]) for i in range(len(param_objs))}

                    bound_plus = qrnn.compiled.assign_parameters(mapping_plus)
                    jobp = sim.run(bound_plus, shots=int(qrnn.shots), seed_simulator=seed)
                    resp = jobp.result()
                    try:
                        counts_plus = resp.get_counts()
                    except Exception:
                        try:
                            counts_plus = resp.results[0].get_counts()
                        except Exception:
                            counts_plus = {}

                    bound_minus = qrnn.compiled.assign_parameters(mapping_minus)
                    jobm = sim.run(bound_minus, shots=int(qrnn.shots), seed_simulator=seed)
                    resm = jobm.result()
                    try:
                        counts_minus = resm.get_counts()
                    except Exception:
                        try:
                            counts_minus = resm.results[0].get_counts()
                        except Exception:
                            counts_minus = {}

                    probs_plus = counts_to_probs(counts_plus)
                    probs_minus = counts_to_probs(counts_minus)

                    s_plus = float(np.sum(gout * probs_plus))
                    s_minus = float(np.sum(gout * probs_minus))
                    diff = s_plus - s_minus

                    contrib = (diff / (2.0 * c)) * (1.0 / delta)
                    g_est += contrib

                g_est /= float(m)
                all_dL_dparam[b] = g_est

            for b in range(batch_size):
                dL_dparam = all_dL_dparam[b]
                grad_encoded = dL_dparam.reshape((time_steps, params_per_timestep))
                device = x_saved.device
                dtype = x_saved.dtype
                grad_encoded_torch = torch.tensor(grad_encoded, dtype=dtype, device=device)
                grad_x[b] = grad_encoded_torch

            return (None, grad_x, None, None, None)

        # --- Manual finite-difference (central difference) implementation ---
        if grad_method.startswith("finite") or grad_method.startswith("fd") or grad_method.startswith("finite-diff"):
            h = float(eps)

            # flatten base parameter vectors per sample
            batch_param_values = []
            for b in range(batch_size):
                flat = []
                for t in range(time_steps):
                    pv = qrnn.param_vectors[t]
                    vals = x_saved[b, t].detach().cpu().numpy().ravel()
                    for i, _ in enumerate(pv):
                        flat.append(float(vals[i]))
                batch_param_values.append(np.array(flat, dtype=float))

            # container for gradients per sample
            all_dL_dparam = np.zeros((batch_size, total_params), dtype=float)

            # For each sample, for each parameter, perturb only that parameter ±h and evaluate
            for b in range(batch_size):
                base = batch_param_values[b]
                gout = grad_output[b].detach().cpu().numpy()  # (time_steps, num_outcomes)
                g_param = np.zeros((total_params,), dtype=float)

                for j in range(total_params):
                    # p + h on parameter j
                    p_plus = base.copy()
                    p_minus = base.copy()
                    p_plus[j] += h
                    p_minus[j] -= h

                    mapping_plus = {param_objs[i]: float(p_plus[i]) for i in range(len(param_objs))}
                    mapping_minus = {param_objs[i]: float(p_minus[i]) for i in range(len(param_objs))}

                    bound_plus = qrnn.compiled.assign_parameters(mapping_plus)
                    jobp = sim.run(bound_plus, shots=int(qrnn.shots), seed_simulator=seed)
                    resp = jobp.result()
                    try:
                        counts_plus = resp.get_counts()
                    except Exception:
                        try:
                            counts_plus = resp.results[0].get_counts()
                        except Exception:
                            counts_plus = {}

                    bound_minus = qrnn.compiled.assign_parameters(mapping_minus)
                    jobm = sim.run(bound_minus, shots=int(qrnn.shots), seed_simulator=seed)
                    resm = jobm.result()
                    try:
                        counts_minus = resm.get_counts()
                    except Exception:
                        try:
                            counts_minus = resm.results[0].get_counts()
                        except Exception:
                            counts_minus = {}

                    probs_plus = counts_to_probs(counts_plus)
                    probs_minus = counts_to_probs(counts_minus)

                    s_plus = float(np.sum(gout * probs_plus))
                    s_minus = float(np.sum(gout * probs_minus))

                    # central difference
                    derivative = (s_plus - s_minus) / (2.0 * h)
                    g_param[j] = derivative

                all_dL_dparam[b] = g_param

            for b in range(batch_size):
                dL_dparam = all_dL_dparam[b]
                grad_encoded = dL_dparam.reshape((time_steps, params_per_timestep))
                device = x_saved.device
                dtype = x_saved.dtype
                grad_encoded_torch = torch.tensor(grad_encoded, dtype=dtype, device=device)
                grad_x[b] = grad_encoded_torch

            return (None, grad_x, None, None, None)

        # If unknown method, return zeros
        return (None, grad_x, None, None, None)
