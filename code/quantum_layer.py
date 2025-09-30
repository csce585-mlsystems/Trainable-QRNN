import torch
from torch.autograd import Function
import numpy as np


class QuantumLayer(Function):
    @staticmethod
    def forward(ctx, qrnn_module, x, grad_method="finite-diff", epsilon=1e-2, spsa_samples=1):
        """
        Forward pass through the quantum circuit.
        """
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
        """
        Backward pass with either finite-difference or SPSA gradient estimation.
        """
        (x,) = ctx.saved_tensors
        qrnn = ctx.qrnn
        eps = ctx.epsilon
        grad_method = ctx.grad_method
        spsa_samples = ctx.spsa_samples

        grads = {name: torch.zeros_like(param) for name, param in qrnn.named_parameters()}

        if grad_method == "finite-diff":
            for name, param in qrnn.named_parameters():
                if not param.requires_grad:
                    continue
                for idx in np.ndindex(param.shape):
                    shift = torch.zeros_like(param)
                    shift[idx] = eps

                    param_plus = (param + shift).detach()
                    param_minus = (param - shift).detach()

                    qrnn._set_param(name, param_plus)
                    out_plus = qrnn._quantum_forward(x)

                    qrnn._set_param(name, param_minus)
                    out_minus = qrnn._quantum_forward(x)

                    grads[name][idx] = ((out_plus - out_minus) / (2 * eps) * grad_output).sum()

                qrnn._restore_params()

        elif grad_method == "spsa":
            for name, param in qrnn.named_parameters():
                if not param.requires_grad:
                    continue
                grad_est = torch.zeros_like(param)
                for _ in range(spsa_samples):
                    delta = torch.randint_like(param, low=0, high=2) * 2 - 1  # ±1
                    param_plus = (param + eps * delta).detach()
                    param_minus = (param - eps * delta).detach()

                    qrnn._set_param(name, param_plus)
                    out_plus = qrnn._quantum_forward(x)

                    qrnn._set_param(name, param_minus)
                    out_minus = qrnn._quantum_forward(x)

                    grad_est += ((out_plus - out_minus) / (2 * eps)) * delta

                grad_est /= spsa_samples
                grads[name] = (grad_est * grad_output).sum()

                qrnn._restore_params()

        grad_list = []
        for name, param in qrnn.named_parameters():
            grad_list.append(grads.get(name, None))

        return (None, None, None, None, None)
