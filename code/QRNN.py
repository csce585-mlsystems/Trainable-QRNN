import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
import numpy as np
import ReuploadingBlock
from qiskit_aer import AerSimulator


class QRNN(nn.Module):
    def __init__(self,n_qubits,repeat_blocks,in_dim,out_dim,context_length,sequence_length,batch_size,shots=4096):
        super(QRNN, self).__init__()
        self.n_qubits = n_qubits
        self.repeat_blocks = repeat_blocks
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.context_length = context_length
        self.shots = shots
        self.sequence_length = sequence_length
        self.batch_size = batch_size

        assert n_qubits % 2 == 0, "n_qubits must be even"

        #Initialize register names
        n_readout = self.n_qubits // 2
        n_memory = self.n_qubits - n_readout

        self.memory_qubits = [f'm{i+1}' for i in range(n_memory)]
        self.readout_qubits = [f'r{i+1}' for i in range(n_readout)]
        self.qubits = []

        for i in range(n_readout):
            self.qubits.append(self.memory_qubits[i])
            self.qubits.append(self.readout_qubits[i])

        # If n_qubits is odd, add the final memory qubit
        if self.n_qubits % 2 != 0:
            self.qubits.append(self.memory_qubits[-1])

        #Create device
        dev = qml.device(name='lightning.qubit', batch_obs=True,wires=self.qubits)

        self.params_c = ((self.n_qubits*3*3)+(n_qubits//2)*2+(n_qubits-1))

        self.input_layer = nn.Sequential(
            nn.Linear(self.in_dim*self.context_length, self.params_c), #Note, this only works for even numbered n_qubits
        )

        #for p in self.input_layer.parameters():
        #    p.requires_grad = False
        
        #Weight shapes for torchlayer
        self.init_weights()
        # QRNN Block

        self.quantum_layer = qml.qnn.TorchLayer(qml.set_shots(qml.QNode(ReuploadingBlock.ReuploadingBlock(self.qubits,self.memory_qubits,self.readout_qubits,self.repeat_blocks,self.sequence_length,self.batch_size,self.params_c),device=dev,interface='torch',mcm_method="one-shot",postselect_mode='fill-shots',diff_method='spsa'), shots=shots),self.weight_shapes,
                                                 init_method=self.init_method)
        #Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(2**(n_readout), self.out_dim),
        )
        
    def init_weights(self):

        #self.CONSTANT = (self.context_length*(self.repeat_blocks)*np.ceil(self.dim))
        self.weight_shapes = {
            "W_bias": (self.n_qubits, 3),
            "W_hidden": (self.n_qubits // 2, 2),
            "W_entangle": (self.n_qubits-1,),
        }


        self.W_bias = torch.empty(self.n_qubits, 3).uniform_((-np.pi / 2) / self.repeat_blocks, (np.pi / 2) / self.repeat_blocks).requires_grad_(requires_grad=True)
        self.W_hidden = torch.empty(self.n_qubits // 2, 2).uniform_((-np.pi / 2) / self.repeat_blocks, (np.pi / 2) / self.repeat_blocks).requires_grad_(requires_grad=True)
        self.W_entangle = torch.empty(self.n_qubits-1).uniform_((-np.pi / 2) / self.repeat_blocks, (np.pi / 2) / self.repeat_blocks).requires_grad_(requires_grad=True)
        self.W_bias    = nn.Parameter(self.W_bias)
        self.W_hidden  = nn.Parameter(self.W_hidden)
        self.W_entangle= nn.Parameter(self.W_entangle)
        self.init_method = {
            "W_bias": self.W_bias,
            "W_hidden": self.W_hidden,
            "W_entangle": self.W_entangle,
        }

    def forward(self, x):
        x = self.input_layer(x)
 
        #x = torch.reshape(x,(self.batch_size,-1))
        #x.flatten()
        quantum_out = self.quantum_layer(x)
        #print(quantum_out.shape)
        quantum_out = torch.reshape(quantum_out,(-1,int(2**(self.n_qubits//2))))

        #print(quantum_out.shape)
        final = self.output_layer(quantum_out)
        return final
    
    def forward_infer(self,x):
        inputs = torch.empty(x.shape[0],self.params_c)

        for i in range(x.shape[0]):
            inputs = torch.cat((inputs,self.input_layer(x[0])),dim=0) 
            
        print(inputs.shape)
    

