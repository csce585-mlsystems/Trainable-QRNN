import numpy as np
import qiskit as qs


def Rot(qc, phi=0, theta=0, omega=0, wires=0):
    qc.rz(phi, wires)
    qc.rx(theta, wires)
    qc.rz(omega, wires)

def encode_input(W_in, x):
    encoded = np.dot(W_in, x)
    phi, theta, omega = encoded
    return phi, theta, omega

def ReuploadingBlock(input_data,
                     n_qubits,
                     context_length,
                     repeat_blocks,
                     W_in,
                     W_bias,
                     W_hidden,
                     W_entangle,
                     seed=0):
    """
    Build the full QRNN circuit with mid-circuit measurement + reset.
    """
    np.random.seed(seed)
    assert n_qubits % 2 == 0, "n_qubits must be even"

    q_mem = qs.circuit.QuantumRegister(n_qubits, "q")
    qc = qs.QuantumCircuit(q_mem, name="QRNN")

    # readout qubits = odd indices
    readout_idx = [i for i in range(n_qubits) if i % 2 == 1]
    register_names = []

    # loop over timesteps
    for t_index in range(input_data.shape[0]):
        # repeat block encoding
        for _ in range(repeat_blocks):
            for k in range(0, n_qubits, 2):
                # encode inputs into three scalars per qubit
                input = input_data[t_index]
                phi, theta, omega = encode_input(W_in[k], input)
                phi2, theta2, omega2 = encode_input(W_in[k + 1], input)

                bias1, bias2, bias3 = W_bias[k]
                bias21, bias22, bias23 = W_bias[k + 1]

                # first rotations
                Rot(qc, phi + bias1, theta + bias2, omega + bias3, wires=k)
                Rot(qc, phi2 + bias21, theta2 + bias22, omega2 + bias23, wires=k + 1)

                qc.cx(k, k + 1)

                # second rotations
                Rot(qc, phi + bias1, theta + bias2, omega + bias3, wires=k)
                Rot(qc, phi2 + bias21, theta2 + bias22, omega2 + bias23, wires=k + 1)

                qc.cry(W_hidden[k // 2, 0], k, k + 1)

                # third rotations
                Rot(qc, phi + bias1, theta + bias2, omega + bias3, wires=k)
                Rot(qc, phi2 + bias21, theta2 + bias22, omega2 + bias23, wires=k + 1)

                qc.crx(W_hidden[k // 2, 1], k, k + 1)

            # entangling layer
            for k in range(0, n_qubits, 2):
                qc.crz(W_entangle[k // 2], k, (k + 2) % n_qubits)

        # classical register for this timestep
        qc.barrier()
        reg_name = f"cr_{t_index}"
        register_names.append(reg_name)
        c_reg = qs.circuit.ClassicalRegister(n_qubits // 2, reg_name)
        qc.add_register(c_reg)

        # mid-circuit measure + reset
        qc.measure(readout_idx, c_reg)
        qc.reset(readout_idx)
        qc.barrier()

    return qc, register_names
