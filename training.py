import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from cnn import SimpleRamanCNN

# 1. Load data
X = np.load('data/X_reference.npy')
y = np.load('data/y_reference.npy')

# 2. Indices
indices = np.arange(len(X))
np.random.seed(42) 
np.random.shuffle(indices)

X = X[indices]
y = y[indices]

# 3. Training (90%) and Test (10%) data 
X_train = X[:54000]
y_train = y[:54000]
X_test_split = X[54000:]
y_test_split = y[54000:]

np.save('data/X_train_split.npy', X_train)
np.save('data/y_train_split.npy', y_train)
np.save('data/X_test_split.npy', X_test_split)
np.save('data/y_test_split.npy', y_test_split)

# 4. Transform to Tensors and Z-normalization
X_train = torch.from_numpy(X_train).float()
X_train = torch.stack([(e - e.mean()) / e.std() for e in X_train])
X_train = X_train.unsqueeze(1) 

y_train = torch.from_numpy(y_train).long()

# Configuración del cargador de datos
batch_size = 64
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

# 2. Initialize model and  optimize
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleRamanCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.NLLLoss()

# 3. Training loop
epochs = 50 
trained_model_path = "simple_raman_cnn.pth"

model.train()
for epoch in range(epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Reset de gradientes
        optimizer.zero_grad()
        
        # Forward pass + Loss
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backpropagation + Update
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 100 == 99:
            print(f"Epoch {epoch+1}, Batch {i+1}: Loss = {running_loss/100:.4f}")
            running_loss = 0.0

# 4. Save model with new weights 
torch.save(model.state_dict(), trained_model_path)

print(f"Training completed. Model saved in {trained_model_path}")

