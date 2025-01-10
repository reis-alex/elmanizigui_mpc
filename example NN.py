import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Generate Dataset for f(x) = x^2
np.random.seed(42)
#x_data = np.random.uniform(0, 2*np.pi, 100).astype(np.float32)  # Random input points
x_data = np.linspace(0,2*np.pi,2000).astype(np.float32)
y_data = (np.sin(x_data)*x_data**2).astype(np.float32)                 # True function output

# Convert to PyTorch tensors
x_train = torch.tensor(x_data).unsqueeze(1)  # Shape: (100, 1)
y_train = torch.tensor(y_data).unsqueeze(1)  # Shape: (100, 1)

# Define Neural Network Model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.hidden = nn.Linear(1, 5)  # Input layer to hidden layer
        self.relu = nn.ReLU()           # ReLU activation
        self.sigmoid = nn.Sigmoid();
        self.output = nn.Linear(5, 1)  # Hidden layer to output layer

    def forward(self, x):
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.relu(x)
        x = self.output(x)
        return x

# Instantiate the model, define loss and optimizer
model = NeuralNetwork()
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.Adam(model.parameters(), lr=0.1)  # Adam Optimizer

# Training Loop
epochs = 15000
loss_history = []

for epoch in range(epochs):
    # Forward pass
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Record loss
    loss_history.append(loss.item())
    
    # Print loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Test the model
x_test = torch.linspace(0, 2*np.pi, 2000).unsqueeze(1)  # Test points
y_test_pred = model(x_test).detach().numpy()      # Predictions

# Plot Results
plt.figure(figsize=(8, 6))
plt.scatter(x_data, y_data, label='True Function (x^2)', color='blue')
plt.plot(x_test.numpy(), y_test_pred, label='Neural Network Prediction', color='red')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Neural Network Approximation of f(x) = x^2')
plt.legend()
plt.grid()
plt.show()