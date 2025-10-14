import numpy as np
import qiskit as qs
from qiskit.circuit import ParameterVector

def Rot(qc, phi, theta, omega, wire):
    qc.rz(phi, wire)
    qc.rx(theta, wire)
    qc.rz(omega, wire)

def ReuploadingBlock(n_qubits, context_length, repeat_blocks, timesteps, seed=0):
    np.random.seed(seed)
    assert n_qubits % 2 == 0
    q_mem = qs.QuantumRegister(n_qubits, "q")
    qc = qs.QuantumCircuit(q_mem, name="QRNN")
    readout_idx = [i for i in range(n_qubits) if i % 2 == 1]
    params_per_timestep = ((n_qubits*3*3)+(n_qubits//2)*2+(n_qubits-1))#4 * n_qubits + (n_qubits - 1)
    print(f"Params per timestep: {params_per_timestep}")
    param_vectors = []
    register_names = []
    for t in range(timesteps):
        pv = ParameterVector(f"x_{t}", params_per_timestep)
        param_vectors.append(pv)
        # qc.h(0)
        # for i in range(0, n_qubits - 1):
        #     qc.cx(i, i + 1)
        # for i in range(n_qubits):
        #     qc.h(i)
    for t_index, pv in enumerate(param_vectors):
        for _ in range(repeat_blocks):
            index = 0
            for k in range(0, n_qubits, 2):
                phi = pv[index]
                theta = pv[index + 1]
                omega = pv[index + 2]
                phi2 = pv[index + 3]
                theta2 = pv[index + 4]
                omega2 = pv[index + 5]
                Rot(qc, phi, theta, omega, k)
                Rot(qc, phi2, theta2, omega2, k + 1)
                qc.cx(k, k + 1)
                phi = pv[index + 6]
                theta = pv[index + 7]
                omega = pv[index + 8]
                phi2 = pv[index + 9]
                theta2 = pv[index + 10]
                omega2 = pv[index + 11]
                Rot(qc, phi, theta, omega, k)
                Rot(qc, phi2, theta2, omega2, k + 1)
                qc.cry(pv[index + 12], k, k + 1)
                phi = pv[index + 13]
                theta = pv[index + 14]
                omega = pv[index + 15]
                phi2 = pv[index + 16]
                theta2 = pv[index + 17]
                omega2 = pv[index + 18]
                Rot(qc, phi, theta, omega, k)
                Rot(qc, phi2, theta2, omega2, k + 1)
                qc.crx(pv[index + 19], k, k + 1)
                index += 20
            for k in range(0, n_qubits-1):
                qc.crz(pv[index], k, (k + 1) % n_qubits)
                index += 1
        qc.barrier()
        reg_name = f"cr_{t_index}"
        register_names.append(reg_name)
        c_reg = qs.circuit.ClassicalRegister(len(readout_idx), reg_name)
        qc.add_register(c_reg)
        qc.measure(readout_idx, c_reg)
        qc.reset(readout_idx)
        qc.barrier()
    return qc, param_vectors, register_names
