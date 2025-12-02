
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from QRNN import QRNN
from tqdm import tqdm
import time

def create_test_data(CONTEXT_LENGTH, PREDICTION_HORIZON, IN_DIM, OUT_DIM, split, device):
    try:
        raw_data = np.load("train_data_lorenz_full.npy")
    except FileNotFoundError:
        print("Error: 'train_data_lorenz_full.npy' not found.")
        raise

    # Split into train/test
    split_index = int(len(raw_data) * split)
    test_data = raw_data[split_index:]  # full test sequence

    # Input = only the driving dimension (col 0)
    # Labels = the prediction targets (col 1,2)
    X_test = torch.tensor(test_data[:, 0], dtype=torch.float32).view(1, -1, 1).to(device)  # shape (1, seq_len, in_dim)
    y_test = torch.tensor(test_data[:, 1:3], dtype=torch.float32).view(1, -1, OUT_DIM).to(device)

    # --- Build context-windowed test sequence ---
    inputs = []
    labels = []
    total_length = CONTEXT_LENGTH + PREDICTION_HORIZON

    for i in range(len(test_data) - total_length + 1):
        # input = sliding window from first column
        window = test_data[i : i + CONTEXT_LENGTH, 0]
        inputs.append(window)

        # label = future target from columns 1:3
        label = test_data[i + CONTEXT_LENGTH + PREDICTION_HORIZON - 1, 1:3]
        labels.append(label)

    inputs = inputs
    labels = labels

    X_test = torch.tensor(inputs, dtype=torch.float32).view(1, -1, CONTEXT_LENGTH * IN_DIM).to(device)
    y_test = torch.tensor(labels, dtype=torch.float32).view(1, -1, OUT_DIM).to(device)

    print(f"Test data shape: {X_test.shape}, Labels shape: {y_test.shape}")
    return X_test, y_test

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
REPEAT_BLOCKS = 1
CONTEXT_LENGTH = 1
SEQUENCE_LENGTH = 20
PREDICTION_HORIZON = 1
IN_DIM = 1
OUT_DIM = 2
SPSA_SAMPLES = 2
SPSA_EPS = .1
GPU = False
DIFF_METHOD = "spsa-w"
SEED = 1929#np.random.randint(1,10000)

LOAD_CHECKPOINT = False
checkpoint_path = "./checkpoints/lorenz_8_finite-diff-w_SIMPLE_1_1024_2_QRNN_1385_LAST.pth"


SHOTS = 1024
TRAIN_TEST_SPLIT_RATIO = 0.7

EPOCHS = 10
BATCH_SIZE = 1
LEARNING_RATE = .007

# --- 2. Data Loading and Preparation ---
print("🚀 Starting data preparation...")
if torch.cuda.is_available():
    device = torch.device("cpu") #Using CPU is not faster for the classical layers yet
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

#Splitting data into training and testing sets ---
split_index = int(len(X) * TRAIN_TEST_SPLIT_RATIO)
X_train = X[:split_index]#, X[split_index:]
y_train = y[:split_index]#, y[split_index:]

#Subsample the train data set getting every other sequence
X_train = X_train[::5]
y_train = y_train[::5]

print(f"Training set size: {len(X_train)} sequences")
#print(f"Test set size: {len(X_test)} sequences")

# --- Create DataLoaders for both sets ---
train_dataset = TensorDataset(X_train, y_train)
X_test, y_test = create_test_data(CONTEXT_LENGTH, PREDICTION_HORIZON, IN_DIM, OUT_DIM, TRAIN_TEST_SPLIT_RATIO, device)



train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


# --- 3. Model, Optimizer, and Loss Function ---
print("\n Initializing model...")

model = QRNN(n_qubits=N_QUBITS, repeat_blocks=REPEAT_BLOCKS, in_dim=IN_DIM, out_dim=OUT_DIM,
             context_length=CONTEXT_LENGTH, sequence_length=SEQUENCE_LENGTH, batch_size=BATCH_SIZE,
             grad_method=DIFF_METHOD, shots=SHOTS, seed=SEED, spsa_samples=SPSA_SAMPLES, epsilon=SPSA_EPS, gpu=GPU).to(device)

if LOAD_CHECKPOINT:
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

#Set layer wise learning rates
optimizer = optim.Adam([
    {'params': model.input_layer.parameters(), 'lr': LEARNING_RATE},
    {'params': model.output_layer.parameters(), 'lr': LEARNING_RATE}
], lr=LEARNING_RATE)
#Print the learning rates
for param_group in optimizer.param_groups:
    print(f"Learning rate: {param_group['lr']}")
criterion = nn.MSELoss().to(device)

print("Model, optimizer, and loss function are ready.")

# --- 4. Training Loop ---
print("Starting training...")
losses = []
val_losses = []
for epoch in range(0,EPOCHS):
    model.train()
    epoch_loss = 0.0
    i = 1
    loss_batch = []
    #Print input layer weights
    print("Input layer weights:", model.input_layer.weight)
    print("Input layer bias:", model.input_layer.bias)
    
    for batch_idx, (input_seq, target_seq) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}"):
        start_time = time.time()
        input_seq = input_seq.to(device)
        target_seq = target_seq.to(device)
        optimizer.zero_grad()

        predicted_sequence, quantum_probs = model(input_seq)   # (batch, seq_len, out_dim)
        #print(quantum_probs)
        loss = criterion(predicted_sequence[:, 5:, :], target_seq[:, 5:, :]) #Ignore first timestep
        loss.backward()
        optimizer.step()

        #Print input layer weights
        #print("Input layer weights:", model.input_layer.bias)

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
        
        if i % 100 == 0:
            torch.save(model.state_dict(), f'./checkpoints/lorenz_{N_QUBITS}_{DIFF_METHOD}_SIMPLE_{REPEAT_BLOCKS}_{SHOTS}_{epoch+1}_QRNN_{i}.pth')
        
        
        end_time = time.time()
        #print(f"Iteration {i}, Loss (RMSE): {np.sqrt(loss.item()):.6f}, Time: {end_time - start_time:.6f}s")
        rmse = np.sqrt(loss.item())
        #print(f"Iteration {i}, Loss (RMSE): {rmse:.6f}")
        loss_batch.append(rmse)
        if i % 10 == 0:
            avg_loss = np.array(loss_batch).mean()
            losses.append(avg_loss)
            loss_batch = []
            print(f"Epoch {epoch+1}--- Average Loss over last 10 data points: {avg_loss:.6f} ---")
            np.save('losses.npy', losses)
        i += 1
        
    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Average Training Loss: {avg_epoch_loss:.6f}")
    last_checkpoint_path = f'./checkpoints/lorenz_{N_QUBITS}_{DIFF_METHOD}_SIMPLE_{REPEAT_BLOCKS}_{SHOTS}_{epoch+1}_QRNN_{i}_LAST.pth'
    torch.save(model.state_dict(), last_checkpoint_path)
    #Run validation after each epoch
    #Instantiate new model with full sequence length for evaluation
    eval_model = QRNN(n_qubits=N_QUBITS, repeat_blocks=REPEAT_BLOCKS, in_dim=IN_DIM, out_dim=OUT_DIM,
             context_length=CONTEXT_LENGTH, sequence_length=X_test.shape[1], batch_size=1,
             grad_method=DIFF_METHOD, shots=SHOTS, seed=SEED, spsa_samples=SPSA_SAMPLES, epsilon=SPSA_EPS, gpu=GPU).to(device)
    eval_model.load_state_dict(torch.load(last_checkpoint_path, map_location=device))
    eval_model.eval()
    with torch.no_grad():
        preds,_ = eval_model(X_test)

    preds = preds.cpu().numpy()[0]   # (seq_len, out_dim)
    y_true = y_test.cpu().numpy()[0]
    rmse_dim0 = np.sqrt(np.mean((y_true[:, 0] - preds[:, 0])**2))
    rmse_dim1 = np.sqrt(np.mean((y_true[:, 1] - preds[:, 1])**2))

    val_losses.append((rmse_dim0, rmse_dim1, (rmse_dim0+rmse_dim1)/2))
    print(f"Validation RMSE - Y: {rmse_dim0:.6f}, Z: {rmse_dim1:.6f}, Avg: {(rmse_dim0+rmse_dim1)/2:.6f}")
    np.save('val_losses.npy', val_losses)



print("\n✅ Training complete!")


