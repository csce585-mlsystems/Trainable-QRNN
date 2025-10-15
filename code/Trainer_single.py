
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from QRNN_new import QRNN
from tqdm import tqdm
import time

def create_sequences(data, context_length, sequence_length, time_step_shift):

    inputs = []
    labels = []

    total_length = context_length + time_step_shift + sequence_length - 1

    for start_idx in range(len(data) - total_length + 1):
        input_seq = []
        label_seq = []

        for seq_idx in range(sequence_length):
            input_window = np.array(data[start_idx + seq_idx : start_idx + seq_idx + context_length,0]).flatten()
            label = np.array(data[start_idx + seq_idx + context_length + time_step_shift - 1,1:3]).flatten()

            input_seq.append(input_window)
            label_seq.append(label)

        inputs.append(input_seq)
        labels.append(label_seq)

    inputs = np.array(inputs)
    labels = np.array(labels)
    print(inputs.shape)
    print(labels.shape)
    return inputs, labels


# --- 1. Hyperparameters ---
N_QUBITS = 8
REPEAT_BLOCKS = 3
CONTEXT_LENGTH = 3
SEQUENCE_LENGTH = 5
PREDICTION_HORIZON = 1
IN_DIM = 1
OUT_DIM = 2

SHOTS = 1024
TRAIN_TEST_SPLIT_RATIO = 0.7

EPOCHS = 10
BATCH_SIZE = 1
LEARNING_RATE = .001

# --- 2. Data Loading and Preparation ---
print("🚀 Starting data preparation...")
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

try:
    raw_data = np.load("train_data_lorenz_full.npy")
except FileNotFoundError:
    print("Error: 'train_data_lorenz.npy' not found. Please ensure the file is in the correct directory.")
    exit()

# Create sequences
X, y = create_sequences(raw_data, CONTEXT_LENGTH,SEQUENCE_LENGTH,PREDICTION_HORIZON)
X = torch.tensor(X,dtype=torch.float)
y = torch.tensor(y,dtype=torch.float)
print(f"Total sequences created: {len(X)}")

# --- ✨ NEW: Splitting data into training and testing sets ---
split_index = int(len(X) * TRAIN_TEST_SPLIT_RATIO)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

#Subsample the train data set getting every other sequence
X_train = X_train[::3]
y_train = y_train[::3]

print(f"Training set size: {len(X_train)} sequences")
print(f"Test set size: {len(X_test)} sequences")

# --- Create DataLoaders for both sets ---
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)



train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True) # No shuffle for test set

# --- 3. Model, Optimizer, and Loss Function ---
print("\n🔧 Initializing model...")

model = QRNN(n_qubits=N_QUBITS, repeat_blocks=REPEAT_BLOCKS, in_dim=IN_DIM, out_dim=OUT_DIM,
             context_length=CONTEXT_LENGTH, sequence_length=SEQUENCE_LENGTH, batch_size=BATCH_SIZE,
             grad_method="spsa", shots=SHOTS).to(device)


optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss().to(device)

print("Model, optimizer, and loss function are ready.")

# --- 4. Training Loop ---
print("Starting training...")
losses = []
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    i = 1
    loss_batch = []
    
    for batch_idx, (input_seq, target_seq) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}"):
        start_time = time.time()
        input_seq = input_seq.to(device)
        target_seq = target_seq.to(device)
        optimizer.zero_grad()

        predicted_sequence, quantum_probs = model(input_seq)   # (batch, seq_len, out_dim)
        #print(quantum_probs)
        loss = criterion(predicted_sequence[:, 1:, :], target_seq[:, 1:, :]) #Ignore first timestep
        loss.backward()
        optimizer.step()

        # -----------------------------------------------
        # before step
        # param_ids_before = {n: id(p) for n,p in model.named_parameters()}
        # param_vals_before = {n: p.detach().cpu().clone() for n,p in model.named_parameters()}


        # after step
        # for n,p in model.named_parameters():
        #     print(n, "id-before", param_ids_before[n], "id-after", id(p))
        #     print("norm before", param_vals_before[n].norm().item(), "after", p.detach().cpu().norm().item())
        # for n,p in model.named_parameters():
        #     print(n, p.grad is None, torch.norm(p.grad) if p.grad is not None else None)

        epoch_loss += np.sqrt(loss.item())
        #losses.append(np.sqrt(loss.item()))
        
        if i % 1000 == 0:
            torch.save(model.state_dict(), f'./checkpoints/{epoch+1}_QRNN_{i}.pth')
        
        
        end_time = time.time()
        #print(f"Iteration {i}, Loss (RMSE): {np.sqrt(loss.item()):.6f}, Time: {end_time - start_time:.6f}s")
        rmse = np.sqrt(loss.item())
        #print(f"Iteration {i}, Loss (RMSE): {rmse:.6f}")
        loss_batch.append(rmse)
        if i % 50 == 0:
            avg_loss = np.array(loss_batch).mean()
            losses.append(avg_loss)
            loss_batch = []
            print(f"Epoch {epoch+1}--- Average Loss over last 50 data points: {avg_loss:.6f} ---")
            np.save('losses.npy', losses)
        i += 1
        
    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Average Training Loss: {avg_epoch_loss:.6f}")

print("\n✅ Training complete!")


