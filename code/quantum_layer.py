import torch
from torch.autograd import Function
import numpy as np


class QuantumLayer(Function):
    @staticmethod
    def forward(ctx, qrnn_module, x, grad_method="finite-diff", epsilon=1e-1, spsa_samples=10):
        ctx.qrnn = qrnn_module
        ctx.grad_method = grad_method
        ctx.epsilon = epsilon
        ctx.spsa_samples = spsa_samples

        with torch.no_grad():
            out = qrnn_module._quantum_forward(x)

        ctx.save_for_backward(x)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        qrnn = ctx.qrnn
        eps = ctx.epsilon
        grad_method = ctx.grad_method
        spsa_samples = ctx.spsa_samples

        # param map
        param_map = dict(qrnn.named_parameters())

        # collect trainable params
        grads = {}
        param_names = []
        param_shapes = []
        param_sizes = []
        for name, param in param_map.items():
            if not param.requires_grad:
                continue
            grads[name] = torch.zeros_like(param, dtype=param.dtype, device=param.device)
            param_names.append(name)
            param_shapes.append(tuple(param.shape))
            size = int(np.prod(param_shapes[-1])) if len(param_shapes[-1]) > 0 else 1
            param_sizes.append(size)

        # backup params once
        qrnn._backup()

        if grad_method == "finite-diff":
            # finite-difference over each scalar parameter (VERY SLOW for many params)
            for name in param_names:
                p = param_map[name]
                for idx in np.ndindex(p.shape):
                    # build shift
                    shift = torch.zeros_like(p)
                    shift[idx] = eps
                    p_plus = (p + shift).detach()
                    p_minus = (p - shift).detach()

                    qrnn._set_param(name, p_plus)
                    out_plus = qrnn._quantum_forward(x)

                    qrnn._set_param(name, p_minus)
                    out_minus = qrnn._quantum_forward(x)

                    # proper chain rule: sum over outputs (not mean)
                    # result is a scalar
                    rho = ((out_plus - out_minus) / (2.0 * eps) * grad_output).sum()

                    # assign scalar into grads
                    grads[name][idx] = rho.to(grads[name].dtype).to(grads[name].device)

                # restore all original params before moving to next param
                qrnn._restore_params()

        elif grad_method == "spsa":
            total_params = sum(param_sizes)
            if total_params > 0:
                grad_est_map = {name: torch.zeros_like(param_map[name], dtype=param_map[name].dtype, device=param_map[name].device)
                                for name in param_names}

                device_for_rand = next(iter(param_map.values())).device

                for s in range(spsa_samples):
                    flat_delta = (torch.randint(0, 2, (total_params,), device=device_for_rand) * 2 - 1).to(torch.float32)

                    # unpack deltas
                    deltas = {}
                    offset = 0
                    for name, size, shape in zip(param_names, param_sizes, param_shapes):
                        part = flat_delta[offset: offset + size].view(shape)
                        p = param_map[name]
                        deltas[name] = part.to(dtype=p.dtype, device=p.device)
                        offset += size

                    # apply +delta to all params
                    for name, delta in deltas.items():
                        p = param_map[name]
                        qrnn._set_param(name, (p + eps * delta).detach())

                    out_plus = qrnn._quantum_forward(x)

                    # apply -delta to all params
                    for name, delta in deltas.items():
                        p = param_map[name]
                        qrnn._set_param(name, (p - eps * delta).detach())

                    out_minus = qrnn._quantum_forward(x)

                    rho = ((out_plus - out_minus) / (2.0 * eps) * grad_output).sum()

                    # accumulate
                    for name, delta in deltas.items():
                        grad_est_map[name] += rho.to(delta.dtype).to(delta.device) * delta

                    qrnn._restore_params()

                for name in param_names:
                    grads[name] = grad_est_map[name] / float(spsa_samples)
        else:
            raise RuntimeError(f"Unknown grad_method: {grad_method}")

        # assign gradients to .grad fields
        for name, g in grads.items():
            if name not in param_map:
                continue
            param_obj = param_map[name]
            if isinstance(g, np.ndarray):
                g = torch.tensor(g, dtype=param_obj.dtype, device=param_obj.device)
            else:
                g = g.to(param_obj.dtype).to(param_obj.device)
            if g.shape != param_obj.shape:
                raise RuntimeError(f"Grad shape mismatch for {name}: {g.shape} vs {param_obj.shape}")
            param_obj.grad = g.clone()

        # gradient w.r.t. input x
        grad_x = torch.zeros_like(x, dtype=x.dtype, device=x.device)

        if grad_method == "spsa":
            grad_est = torch.zeros_like(x, dtype=x.dtype, device=x.device)
            for s in range(spsa_samples):
                delta = (torch.randint(0, 2, x.shape, device=x.device).to(x.dtype) * 2.0 - 1.0)
                out_plus = qrnn._quantum_forward((x + eps * delta).detach())
                out_minus = qrnn._quantum_forward((x - eps * delta).detach())
                dir_deriv = (out_plus - out_minus) / (2.0 * eps)
                rho = (dir_deriv * grad_output).sum()
                grad_est += rho * delta
            grad_x = grad_est / float(spsa_samples)

        elif grad_method == "finite-diff":
            # finite-diff across input elements (very expensive)
            for idx in np.ndindex(x.shape):
                shift = torch.zeros_like(x)
                shift[idx] = eps
                out_plus = qrnn._quantum_forward((x + shift).detach())
                out_minus = qrnn._quantum_forward((x - shift).detach())
                grad_x[idx] = ((out_plus - out_minus) / (2.0 * eps) * grad_output).sum()

        # Ensure restore original params (defensive)
        qrnn._restore_params()

        # return gradients corresponding to forward inputs:
        # (None for module, grad_x for x, None for other args)
        return (None, grad_x, None, None, None)
