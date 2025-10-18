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
        if grad_method == 'spsa':
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
        if grad_method == "finite-diff":
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
        
        if grad_method == 'spsa-w':
            m = max(1, int(spsa_samples))
            c = float(eps)
            in_layer = qrnn.input_layer
            W_param = in_layer.weight
            b_param = in_layer.bias
            params_c, in_features = W_param.shape
            num_weights = params_c * in_features
            has_bias = b_param is not None
            raw_inputs = getattr(qrnn, "_last_raw_inputs", None)
            if raw_inputs is None:
                raise RuntimeError("spsa-w requires qrnn._last_raw_inputs")
            raw_flat = raw_inputs.reshape(batch_size * time_steps, -1)
            device = x_saved.device
            dtype = x_saved.dtype
            raw_flat_torch = raw_flat.to(device=device, dtype=dtype)
            gout_all = [grad_output[b].detach().cpu().numpy() for b in range(batch_size)]
            g_est_acc = np.zeros((batch_size, params_c, in_features), dtype=float)
            for k in range(m):
                delta = rng.choice([-1.0, 1.0], size=(params_c, in_features))
                with torch.no_grad():
                    W_param.data += (c * torch.tensor(delta, dtype=W_param.dtype, device=W_param.device))
                with torch.no_grad():
                    encoded_flat = in_layer(raw_flat_torch)
                encoded = encoded_flat.view(batch_size, time_steps, params_c)
                probs = qrnn._quantum_forward(encoded)
                if isinstance(probs, torch.Tensor):
                    probs_np = probs.detach().cpu().numpy()
                else:
                    probs_np = np.array(probs)
                s_plus = np.zeros((batch_size,), dtype=float)
                for b in range(batch_size):
                    gout = gout_all[b]
                    s_plus[b] = float(np.sum(gout * probs_np[b]))
                with torch.no_grad():
                    W_param.data -= (2.0 * c * torch.tensor(delta, dtype=W_param.dtype, device=W_param.device))
                with torch.no_grad():
                    encoded_flat = in_layer(raw_flat_torch)
                encoded = encoded_flat.view(batch_size, time_steps, params_c)
                probs = qrnn._quantum_forward(encoded)
                if isinstance(probs, torch.Tensor):
                    probs_np = probs.detach().cpu().numpy()
                else:
                    probs_np = np.array(probs)
                s_minus = np.zeros((batch_size,), dtype=float)
                for b in range(batch_size):
                    gout = gout_all[b]
                    s_minus[b] = float(np.sum(gout * probs_np[b]))
                with torch.no_grad():
                    W_param.data += (c * torch.tensor(delta, dtype=W_param.dtype, device=W_param.device))
                diff = (s_plus - s_minus)
                for b in range(batch_size):
                    contrib = (diff[b] / (2.0 * c)) * (1.0 / delta)
                    g_est_acc[b] += contrib
            g_est_acc /= float(m)
            dL_dW = g_est_acc.mean(axis=0)
            if has_bias:
                dL_db = np.zeros((params_c,), dtype=float)
                bvals = b_param.detach().cpu().numpy()
                for i in range(params_c):
                    dL_db[i] = 0.0
                # optional: estimate bias via same SPSA direction on bias-only if desired; leave zeros for now
            if W_param.grad is None:
                W_param.grad = torch.zeros_like(W_param.data)
            W_param.grad[:] = torch.tensor(dL_dW, dtype=W_param.dtype, device=W_param.device)
            if has_bias:
                if b_param.grad is None:
                    b_param.grad = torch.zeros_like(b_param.data)
                b_param.grad[:] = torch.tensor(dL_db, dtype=b_param.dtype, device=b_param.device)
            return (None, None, None, None, None)

        if grad_method == 'finite-diff-w':
            h = float(eps)
            in_layer = qrnn.input_layer
            W_param = in_layer.weight
            b_param = in_layer.bias
            params_c, in_features = W_param.shape
            has_bias = b_param is not None
            raw_inputs = getattr(qrnn, "_last_raw_inputs", None)
            if raw_inputs is None:
                raise RuntimeError("finite-w requires qrnn._last_raw_inputs")
            raw_flat = raw_inputs.reshape(batch_size * time_steps, -1)
            device = x_saved.device
            dtype = x_saved.dtype
            raw_flat_torch = raw_flat.to(device=device, dtype=dtype)
            def compute_s_current():
                with torch.no_grad():
                    encoded_flat = in_layer(raw_flat_torch)
                encoded = encoded_flat.view(batch_size, time_steps, params_c)
                probs = qrnn._quantum_forward(encoded)
                if isinstance(probs, torch.Tensor):
                    probs_np = probs.detach().cpu().numpy()
                else:
                    probs_np = np.array(probs)
                s_per_sample = np.zeros((batch_size,), dtype=float)
                for b in range(batch_size):
                    gout = grad_output[b].detach().cpu().numpy()
                    s_per_sample[b] = float(np.sum(gout * probs_np[b]))
                return s_per_sample
            qrnn._backup()
            dL_dW_accum = np.zeros((batch_size, params_c, in_features), dtype=float)
            dL_db_accum = np.zeros((batch_size, params_c), dtype=float) if has_bias else None
            print(params_c * in_features)
            for i in range(params_c):
                for j in range(in_features):
                    with torch.no_grad():
                        W_param.data[i, j] += h
                    s_plus = compute_s_current()
                    with torch.no_grad():
                        W_param.data[i, j] -= 2.0 * h
                    s_minus = compute_s_current()
                    with torch.no_grad():
                        W_param.data[i, j] += h
                    dvals = (s_plus - s_minus) / (2.0 * h)
                    for b in range(batch_size):
                        dL_dW_accum[b, i, j] = dvals[b]
            if has_bias:
                for i in range(params_c):
                    with torch.no_grad():
                        b_param.data[i] += h
                    s_plus = compute_s_current()
                    with torch.no_grad():
                        b_param.data[i] -= 2.0 * h
                    s_minus = compute_s_current()
                    with torch.no_grad():
                        b_param.data[i] += h
                    dvals = (s_plus - s_minus) / (2.0 * h)
                    for b in range(batch_size):
                        dL_db_accum[b, i] = dvals[b]
            dL_dW = dL_dW_accum.mean(axis=0)
            dL_db = dL_db_accum.mean(axis=0) if has_bias else None
            if W_param.grad is None:
                W_param.grad = torch.zeros_like(W_param.data)
            W_param.grad[:] = torch.tensor(dL_dW, dtype=W_param.dtype, device=W_param.device)
            if has_bias:
                if b_param.grad is None:
                    b_param.grad = torch.zeros_like(b_param.data)
                b_param.grad[:] = torch.tensor(dL_db, dtype=b_param.dtype, device=b_param.device)
            qrnn._restore_params()
            return (None, None, None, None, None)

        # If unknown method, return zeros
        print('Unknown Grad Method, returning Zeroes...')
        return (None, grad_x, None, None, None)
