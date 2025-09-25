import pennylane as qml
import torch
from functools import partial

def Rotate(qubit,phi,theta,omega):
    qml.RZ(phi=phi,wires=qubit)
    qml.RX(phi=theta,wires=qubit)
    qml.RZ(phi=omega,wires=qubit)


def ReuploadingBlock(qubits,mem_qubits,read_qubits,repeat_blocks,sequence_length,batch_size,params_c):
    def circuit(inputs,W_bias,W_hidden,W_entangle):
        #print(inputs.shape)
        
        inputs = torch.reshape(inputs,(-1,params_c))
        #print(inputs.shape)
        measurements = []
        for i in range(inputs.shape[0]):
            for j in range(repeat_blocks):
                index = 0 #Weights are the same in each repeat block
                for k in range(0,len(qubits),2):
                    Rotate(qubits[k],inputs[i,index]+W_bias[k,0],inputs[i,index+1]+W_bias[k,1],inputs[i,index+2]+W_bias[k,2])
                    Rotate(qubits[k+1],inputs[i,index+3]+W_bias[k+1,0],inputs[i,index+4]+W_bias[k+1,1],inputs[i,index+5]+W_bias[k+1,2])
                    qml.CNOT(qubits[k:k+2])
                    Rotate(qubits[k],inputs[i,index+6]+W_bias[k,0],inputs[i,index+7]+W_bias[k,1],inputs[i,index+8]+W_bias[k,2])
                    Rotate(qubits[k+1],inputs[i,index+9]+W_bias[k+1,0],inputs[i,index+10]+W_bias[k+1,1],inputs[i,index+11]+W_bias[k+1,2])
                    qml.CRY(inputs[i,index+12]+W_hidden[k//2,0],wires=qubits[k:k+2])
                    Rotate(qubits[k],inputs[i,index+13]+W_bias[k,0],inputs[i,index+14]+W_bias[k,1],inputs[i,index+15]+W_bias[k,2])
                    Rotate(qubits[k+1],inputs[i,index+16]+W_bias[k+1,0],inputs[i,index+17]+W_bias[k+1,1],inputs[i,index+18]+W_bias[k+1,2])
                    qml.CRX(inputs[i,index+19]+W_hidden[k//2,1],wires=qubits[k:k+2])
                    index += (20)
                for k in range(0,len(qubits)-1):
                    qml.CRZ(inputs[i,index]+W_entangle[k],wires=qubits[k:k+2])
                    index+=1

            meas = []
            for qubit in read_qubits:
                meas.append(qml.measure(qubit,reset=True))
            measurements.append(meas)
            
        results = []
        for i in range(inputs.shape[0]):
            results.append(qml.probs(op=measurements[i]))
        return results

    return circuit
