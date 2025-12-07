# %%
import matplotlib.pyplot as plt
import numpy as np

# %%

#losses1= np.load('./losses_e1.npy')

#losses = np.load('../results/server/fdsa_gpu_losses_combined.npy')
losses = np.load('./code/losses.npy')

plt.figure()
plt.plot(losses)
plt.xlabel('Batch')
plt.ylabel('RMSE')
plt.title('Train RMSE with SPSA Gradient')
#plt.savefig('../results/finite_r1_lr001_oct22/train_rmse_fdsa_diff_r1.pdf')
plt.show()

# %%
#Plot validation loss
plt.figure()
val_losses = np.load('./code/val_losses_spsa_last.npy')
#val_losses = np.load('../results/server/val_losses.npy')
plt.plot(val_losses[:,2])
plt.xlabel('Epoch')
plt.xticks(range(0, len(val_losses[:,2]), 1))
plt.ylabel('Validation RMSE')
plt.title('Validation RMSE with SPSA Gradient')
#plt.savefig('../results/finite_r1_lr001_oct22/val_rmse_fdsa_diff_r1.pdf')
plt.show()

# %%
val_losses

# %%
import matplotlib.pyplot as plt
import numpy as np
times = [1.2,18.0,39.58]

#Plot times as bar graph with labels
plt.figure()
plt.bar(['SPSA (CPU)','FDSA (GPU)','FDSA (CPU)'], times, color=['blue', 'orange', 'green'])
plt.ylabel('Time (s)')
plt.title('Time per Weight Update')
plt.show()

hours_per_epoch = [1,8,14.5]
plt.figure()
plt.bar(['SPSA (CPU)','FDSA (GPU)','FDSA (CPU)'], hours_per_epoch, color=['blue', 'orange', 'green'])
plt.ylabel('Time (hours)')
plt.title('Projected time per Epoch (1384 Sequences)')
plt.show()


