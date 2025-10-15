
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

        # forward uses your proven counts-based _quantum_forward
        with torch.no_grad():
            out = qrnn_module._quantum_forward(x)

        return out

    @staticmethod
    def _ensure_sampler_and_grad_calc(qrnn, grad_method, epsilon, spsa_samples):
        """
        Ensure qrnn._sampler and qrnn._grad_cache exist and are consistent with requested settings.
        qrnn._grad_cache will be a dict with keys: 'method', 'epsilon', 'spsa_samples', 'grad_calc'
        """
        # ensure sampler exists (shot-based to match QRNN._quantum_forward)
        if not hasattr(qrnn, "_sampler") or qrnn._sampler is None:
            run_options = {"shots": int(qrnn.shots)}
            if hasattr(qrnn, "seed"):
                # some Aer versions accept seed_simulator via run options, but pass seed via options to grad/run below
                run_options["seed_simulator"] = int(qrnn.seed)
            qrnn._sampler = AerSampler(run_options=run_options)

        # ensure grad cache exists and matches requested config
        if not hasattr(qrnn, "_grad_cache") or qrnn._grad_cache is None:
            qrnn._grad_cache = {}

        cache = qrnn._grad_cache
        need_new = (
            cache.get("method") != grad_method
            or cache.get("epsilon") != epsilon
            or cache.get("spsa_samples") != spsa_samples
            or ("grad_calc" not in cache)
        )
        if need_new:
            # create new grad calc
            method = grad_method.lower()
            if method in ("finite-diff", "finite_diff", "finite", "finite-diff"):
                grad_calc = FiniteDiffSamplerGradient(qrnn._sampler, epsilon=epsilon)
            else:
                grad_calc = SPSASamplerGradient(qrnn._sampler, epsilon=epsilon, batch_size=spsa_samples)
            qrnn._grad_cache = {
                "method": grad_method,
                "epsilon": epsilon,
                "spsa_samples": spsa_samples,
                "grad_calc": grad_calc,
            }
        return qrnn._grad_cache["grad_calc"]

    @staticmethod
    def backward(ctx, grad_output):
        qrnn = ctx.qrnn
        grad_method = ctx.grad_method
        eps = ctx.epsilon
        spsa_samples = ctx.spsa_samples
        #print(f"Gradient calculation method: {grad_method}, epsilon: {eps}, SPSA samples: {spsa_samples}")
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

        # Lazily create / reuse sampler and gradient object
        grad_calc = QuantumLayer._ensure_sampler_and_grad_calc(qrnn, grad_method, eps, spsa_samples)

        # Build a parameter_values list for the entire minibatch so we call grad_calc.run once.
        parameter_values_list = []
        for b in range(batch_size):
            param_values = []
            for t in range(time_steps):
                pv = qrnn.param_vectors[t]
                vals = x_saved[b, t].detach().cpu().numpy().ravel()
                for i, _ in enumerate(pv):
                    param_values.append(float(vals[i]))
            parameter_values_list.append(param_values)

        # Run gradient primitive once for the full minibatch
        # Pass options for shots/seed if desired (most primitives accept options kw)
        run_options = {}
        if hasattr(qrnn, "shots"):
            run_options["shots"] = int(qrnn.shots)
        if hasattr(qrnn, "seed"):
            run_options["seed_simulator"] = int(qrnn.seed)
        try:
            #Create list of circuits for batch

            circuits = [qrnn.compiled] * batch_size
            res_job = grad_calc.run(circuits=circuits, parameter_values=parameter_values_list, options=run_options)
            try:
                res = res_job.result()
            except Exception:
                res = res_job
        except TypeError:
            res_job = grad_calc.run(circuits=[qrnn.compiled], parameter_values=parameter_values_list)
            try:
                res = res_job.result()
            except Exception:
                res = res_job

        gradients_per_sample = [None] * batch_size
        if hasattr(res, "gradients"):
            grads_attr = res.gradients
            # grads_attr might be a list indexed by circuit (batch)
            if isinstance(grads_attr, (list, tuple)) and len(grads_attr) == batch_size:
                # each grads_attr[b] is itself the per-parameter gradients for that circuit
                for b in range(batch_size):
                    gradients_per_sample[b] = grads_attr[b]
            else:

                try:
                    if len(grads_attr) == 1 and isinstance(grads_attr[0], (list, tuple)) and len(grads_attr[0]) == total_params:
                        # Possibly only one circuit's gradients returned (unexpected). Try to broadcast it.
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
            # Try to access res.gradients assuming shape: list length total_params, each entry array-like length = number of full outcomes
            # or array-like that can be indexed by [b] for batch.
            if hasattr(res, "gradients"):
                grads_attr = res.gradients
            elif hasattr(res, "gradient"):
                grads_attr = res.gradient
            else:
                grads_attr = res  # best-effort fallback

            gradients_per_sample = [ [None]*total_params for _ in range(batch_size) ]
            try:
                for j in range(total_params):
                    entry = grads_attr[j]
                    # If entry is array-like and has leading batch dimension, split it
                    arr = np.asarray(entry)
                    if arr.ndim > 0 and arr.shape[0] == batch_size:
                        # per-sample arrays
                        for b in range(batch_size):
                            # store the j-th param gradient for sample b
                            gradients_per_sample[b][j] = arr[b]
                    else:
                        # Same gradient for all samples (unlikely), broadcast
                        for b in range(batch_size):
                            gradients_per_sample[b][j] = entry
            except Exception:
                # final fallback: attempt to use res.gradients[0] for all samples
                try:
                    fallback = res.gradients[0]
                    for b in range(batch_size):
                        gradients_per_sample[b] = fallback
                except Exception:
                    # give up gracefully by returning zeros
                    return (None, grad_x, None, None, None)

   
        for b in range(batch_size):
            gradients_list = gradients_per_sample[b]

            grads_per_param = np.zeros((total_params, time_steps, num_outcomes_per_timestep), dtype=float)

            for j in range(total_params):
                grad_entry = gradients_list[j]
                if isinstance(grad_entry, dict):
                    items = grad_entry.items()
                else:
                    # array-like: enumerate
                    try:
                        arr = np.asarray(grad_entry)
                        items = enumerate(arr)
                    except Exception:
                        items = []

                for full_idx, grad_val in items:
                    try:
                        bits = format(int(full_idx), 'b').zfill(total_bits)
                    except Exception:
                        # if full_idx not an int, skip
                        continue
                    registers = [bits[i * n_readout:(i + 1) * n_readout] for i in range(time_steps)]
                    registers = registers[::-1]  # match forward bit-ordering
                    for t, reg_bits in enumerate(registers):
                        idx = int(reg_bits, 2)
                        grads_per_param[j, t, idx] += float(grad_val)

            # compute dL/dparam_j = sum_{t,idx} dL/dprob_{t,idx} * dprob_{t,idx}/dparam_j
            gout = grad_output[b].detach().cpu().numpy()  # shape (time_steps, num_outcomes_per_timestep)
            dL_dparam = np.zeros((total_params,), dtype=float)
            for j in range(total_params):
                dL_dparam[j] = np.sum(gout * grads_per_param[j])

            grad_encoded = dL_dparam.reshape((time_steps, params_per_timestep))

            device = x_saved.device
            dtype = x_saved.dtype
            grad_encoded_torch = torch.tensor(grad_encoded, dtype=dtype, device=device)

            grad_x[b] = grad_encoded_torch
           
        # Return tuple matching forward inputs: (qrnn_module, x, grad_method, epsilon, spsa_samples)
        
        return (None, grad_x, None, None, None)
