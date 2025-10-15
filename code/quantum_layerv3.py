import torch
from torch.autograd import Function
import numpy as np

# Finite-diff fallback (only used when grad_method isn't SPSA)
from qiskit_aer.primitives import Sampler as AerSampler
from qiskit_algorithms.gradients import FiniteDiffSamplerGradient


class QuantumLayer(Function):
    @staticmethod
    def forward(ctx, qrnn_module, x, grad_method="spsa", epsilon=1e-1, spsa_samples=3):

        ctx.qrnn = qrnn_module
        ctx.grad_method = grad_method
        ctx.epsilon = float(epsilon)
        ctx.spsa_samples = int(spsa_samples)
        ctx.save_for_backward(x)

        # Use your proven counts-based quantum forward (QRNN._quantum_forward)
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

        # gradient w.r.t. the encoded activations (same shape as x_saved)
        grad_x = torch.zeros_like(x_saved)

        # helper: parse counts dict -> (time_steps, num_outcomes) probabilities
        def counts_to_probs(counts_dict):
            num_outcomes = 2 ** n_readout
            probs = np.zeros((time_steps, num_outcomes), dtype=float)
            if not counts_dict:
                return probs
            for bitstring, c in counts_dict.items():
                # forward used: registers = bitstring.split(); registers = registers[::-1]
                registers = bitstring.split()
                registers = registers[::-1]
                for t, reg_bits in enumerate(registers):
                    idx = int(reg_bits, 2)
                    probs[t, idx] += c / float(qrnn.shots)
            return probs

        # Fast path: manual sequential SPSA (mirror your manual implementation)
        if grad_method.startswith("spsa"):
            m = max(1, int(spsa_samples))      # number of probe pairs per sample
            c = float(eps)                     # SPSA perturbation magnitude

            # Precompute flattened base parameter vectors (one per sample)
            batch_param_values = []
            for b in range(batch_size):
                flat = []
                for t in range(time_steps):
                    pv = qrnn.param_vectors[t]
                    vals = x_saved[b, t].detach().cpu().numpy().ravel()
                    for i, _ in enumerate(pv):
                        flat.append(float(vals[i]))
                batch_param_values.append(np.array(flat, dtype=float))

            # Parameter objects for binding (order must match compiled.parameters)
            param_objs = list(qrnn.compiled.parameters)
            if len(param_objs) != total_params:
                # If Qiskit ordering or parameter set differs, we still try to use zipped portion
                param_objs = param_objs[:total_params]

            # PRNG for reproducible Rademacher vectors (seed from qrnn if present)
            seed = int(getattr(qrnn, "seed", 0) or 0)
            rng = np.random.default_rng(seed)

            # Container for gradients per sample (flattened)
            all_dL_dparam = np.zeros((batch_size, total_params), dtype=float)

            # For each sample, do sequential SPSA (exact same path as manual)
            sim = qrnn.sim

            for b in range(batch_size):
                base = batch_param_values[b]
                g_est = np.zeros((total_params,), dtype=float)
                gout = grad_output[b].detach().cpu().numpy()  # (time_steps, num_outcomes)

                for k in range(m):
                    # Rademacher perturbation
                    delta = rng.choice([-1.0, 1.0], size=(total_params,))
                    p_plus = base + c * delta
                    p_minus = base - c * delta

                    # build mapping Parameter -> value in compiled order
                    mapping_plus = {param_objs[i]: float(p_plus[i]) for i in range(len(param_objs))}
                    mapping_minus = {param_objs[i]: float(p_minus[i]) for i in range(len(param_objs))}

                    # bind and run sequentially (matching your manual path)
                    bound_plus = qrnn.compiled.assign_parameters(mapping_plus)
                    jobp = sim.run(bound_plus, shots=int(qrnn.shots), seed_simulator=int(getattr(qrnn, "seed", 0) or 0))
                    resp = jobp.result()
                    try:
                        counts_plus = resp.get_counts()
                    except Exception:
                        # fallback: try to access results list
                        try:
                            counts_plus = resp.results[0].get_counts()
                        except Exception:
                            counts_plus = {}

                    bound_minus = qrnn.compiled.assign_parameters(mapping_minus)
                    jobm = sim.run(bound_minus, shots=int(qrnn.shots), seed_simulator=int(getattr(qrnn, "seed", 0) or 0))
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

                    # scalar objective s = sum_{t,idx} gout[t,idx] * prob[t,idx]
                    s_plus = float(np.sum(gout * probs_plus))
                    s_minus = float(np.sum(gout * probs_minus))
                    diff = s_plus - s_minus

                    # SPSA gradient contribution: (s_plus - s_minus) / (2*c*delta_j)
                    # delta_j is ±1 so 1/delta_j is ±1
                    contrib = (diff / (2.0 * c)) * (1.0 / delta)
                    g_est += contrib

                # average over m probes
                g_est /= float(m)
                all_dL_dparam[b] = g_est

            # reshape per-sample flattened grads into (time_steps, params_per_timestep) and pack into grad_x
            for b in range(batch_size):
                dL_dparam = all_dL_dparam[b]
                grad_encoded = dL_dparam.reshape((time_steps, params_per_timestep))
                device = x_saved.device
                dtype = x_saved.dtype
                grad_encoded_torch = torch.tensor(grad_encoded, dtype=dtype, device=device)
                grad_x[b] = grad_encoded_torch

            return (None, grad_x, None, None, None)

        # Fallback: finite-difference using Qiskit's FiniteDiffSamplerGradient (batched)
        # We keep a cached sampler+grad_calc on the qrnn instance to avoid re-instantiation cost
        # Create AerSampler with shots run_options to match your shot-based forward if not present
        if not hasattr(qrnn, "_sampler") or qrnn._sampler is None:
            run_options = {"shots": int(getattr(qrnn, "shots", 1024))}
            qrnn._sampler = AerSampler(run_options=run_options)

        # create or reuse cached FiniteDiffSamplerGradient instance
        if not hasattr(qrnn, "_finite_diff_grad") or qrnn._finite_diff_grad is None or getattr(qrnn, "_finite_diff_eps", None) != eps:
            qrnn._finite_diff_grad = FiniteDiffSamplerGradient(qrnn._sampler, epsilon=eps)
            qrnn._finite_diff_eps = eps

        grad_calc = qrnn._finite_diff_grad

        # Build parameter_values list across batch
        parameter_values_list = []
        for b in range(batch_size):
            flat = []
            for t in range(time_steps):
                pv = qrnn.param_vectors[t]
                vals = x_saved[b, t].detach().cpu().numpy().ravel()
                for i, _ in enumerate(pv):
                    flat.append(float(vals[i]))
            parameter_values_list.append(flat)

        # call grad_calc.run in a batched way: repeat compiled circuit to match parameter vectors length
        circuits = [qrnn.compiled] * batch_size
        try:
            res_job = grad_calc.run(circuits=circuits, parameter_values=parameter_values_list,
                                    options={"shots": int(getattr(qrnn, "shots", 1024)),
                                             "seed_simulator": int(getattr(qrnn, "seed", 0) or 0)})
            try:
                res = res_job.result()
            except Exception:
                res = res_job
        except TypeError:
            # older qiskit version signature fallback
            res_job = grad_calc.run(circuits=circuits, parameter_values=parameter_values_list)
            try:
                res = res_job.result()
            except Exception:
                res = res_job

        # parse res.gradients into per-sample gradient lists (robust parsing)
        gradients_per_sample = [None] * batch_size
        if hasattr(res, "gradients"):
            grads_attr = res.gradients
            if isinstance(grads_attr, (list, tuple)) and len(grads_attr) == batch_size:
                for b in range(batch_size):
                    gradients_per_sample[b] = grads_attr[b]
            else:
                # try to broadcast single-circuit result to all samples
                try:
                    if len(grads_attr) == 1 and isinstance(grads_attr[0], (list, tuple)) and len(grads_attr[0]) == total_params:
                        for b in range(batch_size):
                            gradients_per_sample[b] = grads_attr[0]
                    else:
                        gradients_per_sample = None
                except Exception:
                    gradients_per_sample = None
        elif hasattr(res, "gradient"):
            grads_attr = res.gradient
            if isinstance(grads_attr, (list, tuple)) and len(grads_attr) == batch_size:
                for b in range(batch_size):
                    gradients_per_sample[b] = grads_attr[b]
            else:
                gradients_per_sample = None
        else:
            gradients_per_sample = None

        if gradients_per_sample is None:
            # interpret res.gradients as list of per-param arrays with a leading batch axis
            if hasattr(res, "gradients"):
                grads_attr = res.gradients
            elif hasattr(res, "gradient"):
                grads_attr = res.gradient
            else:
                grads_attr = res

            gradients_per_sample = [ [None]*total_params for _ in range(batch_size) ]
            try:
                for j in range(total_params):
                    entry = grads_attr[j]
                    arr = np.asarray(entry)
                    if arr.ndim > 0 and arr.shape[0] == batch_size:
                        for b in range(batch_size):
                            gradients_per_sample[b][j] = arr[b]
                    else:
                        for b in range(batch_size):
                            gradients_per_sample[b][j] = entry
            except Exception:
                try:
                    fallback = res.gradients[0]
                    for b in range(batch_size):
                        gradients_per_sample[b] = fallback
                except Exception:
                    return (None, grad_x, None, None, None)

        # convert per-sample per-parameter full-outcome derivatives into dL/dparam and then to grad_x
        for b in range(batch_size):
            gradients_list = gradients_per_sample[b]
            grads_per_param = np.zeros((total_params, time_steps, num_outcomes_per_timestep), dtype=float)

            for j in range(total_params):
                grad_entry = gradients_list[j]
                if isinstance(grad_entry, dict):
                    items = grad_entry.items()
                else:
                    try:
                        arr = np.asarray(grad_entry)
                        items = enumerate(arr)
                    except Exception:
                        items = []

                for full_idx, grad_val in items:
                    try:
                        bits = format(int(full_idx), 'b').zfill(total_bits)
                    except Exception:
                        continue
                    registers = [bits[i * n_readout:(i + 1) * n_readout] for i in range(time_steps)]
                    registers = registers[::-1]
                    for t, reg_bits in enumerate(registers):
                        idx = int(reg_bits, 2)
                        grads_per_param[j, t, idx] += float(grad_val)

            gout = grad_output[b].detach().cpu().numpy()
            dL_dparam = np.zeros((total_params,), dtype=float)
            for j in range(total_params):
                dL_dparam[j] = np.sum(gout * grads_per_param[j])

            grad_encoded = dL_dparam.reshape((time_steps, params_per_timestep))
            device = x_saved.device
            dtype = x_saved.dtype
            grad_encoded_torch = torch.tensor(grad_encoded, dtype=dtype, device=device)
            grad_x[b] = grad_encoded_torch

        return (None, grad_x, None, None, None)
