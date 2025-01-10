import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

#x_data1 = np.linspace(-0,10,2000).astype(np.float32)
#x_data2 = np.linspace(-10,0,2000).astype(np.float32)
x_data1 = np.random.uniform(-10, 10, 100).astype(np.float32)  # Random values for x1
x_data2 = np.random.uniform(-10, 10, 100).astype(np.float32)
y_data = (x_data1+x_data2).astype(np.float32)                 # True function output


# Convert to PyTorch tensors
x_train = torch.tensor(np.column_stack((x_data1,x_data2)))  
y_train = torch.tensor(y_data).unsqueeze(1)

# Define Neural Network Model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        #self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 5),
            nn.ReLU(),
            nn.Linear(5, 5),
            nn.ReLU(),
            nn.Linear(5, 1),
        )
        #self.hidden = nn.Linear(2, 5)  # Input layer to hidden layer
        #self.relu = nn.ReLU()           # ReLU activation
        #self.sigmoid = nn.Sigmoid();
        #self.output = nn.Linear(5, 1)  # Hidden layer to output layer

    def forward(self, x):
        #x = self.flatten(x)
        x = self.linear_relu_stack(x)
        return x
        #x = self.hidden(x)
        #x = self.sigmoid(x)
        #x = self.relu(x)
        #x = self.output(x)
        #return x

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
x_test = torch.tensor(np.column_stack((torch.linspace(-20, 20, 2000),torch.linspace(-20, 20, 2000))))  # Test points
y_test_pred = model(x_test).detach().numpy()      # Predictions
y_test_real = (x_test[0:,0].detach().numpy()+x_test[0:,1].detach().numpy()).astype(np.float32) 

# Plot Results
plt.figure(figsize=(8, 6))
plt.scatter(x_data1, y_data, label='True Function (x^2)', color='blue')
plt.plot(x_test[0:,1], y_test_pred, label='Neural Network Prediction', color='red')
plt.plot(x_test[0:,1], y_test_real, label='Neural Network Prediction', color='green')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Neural Network Approximation of f(x) = x^2')
plt.legend()
plt.grid()
plt.show()