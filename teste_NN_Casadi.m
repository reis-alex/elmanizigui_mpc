% Neural Network Training with CasADi in MATLAB
clc;
clear;

import casadi.*

% Define Hyperparameters
input_dim       = 1;       % Number of input features
hidden_dim      = 10;      % Number of hidden neurons
output_dim      = 1;      % Number of output neurons
learning_rate   = .1; % Learning rate
epochs          = 500;       % Number of epochs

% Generate Toy Dataset
X = linspace(-2,2,100);
Y = X.^2;

% Symbolic Variables
x = MX.sym('x', input_dim);     % Input
y_true = MX.sym('y_true', 1);   % True output

% Initialize Weights and Biases
W1 = MX.sym('W1', hidden_dim, input_dim); % Input to hidden weights
b1 = MX.sym('b1', hidden_dim, 1);         % Hidden layer biases
W2 = MX.sym('W2', hidden_dim, hidden_dim); % Hidden to output weights
b2 = MX.sym('b2', hidden_dim, 1);         % Output layer biases
W3 = MX.sym('W2', output_dim, hidden_dim); % Hidden to output weights
b3 = MX.sym('b2', output_dim, 1);         % Output layer biases

% Define Neural Network
relu = @(z) max(0, z); % ReLU activation
sigmoid = @(z) 1 ./ (1 + exp(-z)); % Sigmoid activation

% Forward Pass
z1 = W1 * x + b1;
a1 = sigmoid(z1);
z2 = W2 * a1 + b2;
a2 = sigmoid(z2);
z3 = W3*a2 + b3;
y_pred = relu(z3);

% Loss Function (Binary Cross-Entropy)
% loss = -y_true * log(y_pred) - (1 - y_true) * log(1 - y_pred);
loss = (y_true-y_pred).^2/100;
% Gradients Using CasADi
params = {W1, b1, W2, b2}; % Parameters to optimize
grads = gradient(loss, [W1(:); b1(:); W2(:); b2(:); W3(:); b3(:)]);

% CasADi Function for Forward and Backward Pass
f_loss = Function('f_loss', {x, y_true, W1, b1, W2, b2, W3, b3}, {loss});
f_grad = Function('f_grad', {x, y_true, W1, b1, W2, b2, W3, b3}, {grads});

% Initialize Parameters
W1_val = randn(hidden_dim, input_dim) * 0.1;
b1_val = randn(hidden_dim, 1) * 0.1;
W2_val = randn(hidden_dim, hidden_dim) * 0.1;
b2_val = randn(hidden_dim, 1) * 0.1;
W3_val = randn(output_dim, hidden_dim) * 0.1;
b3_val = randn(output_dim, 1) * 0.1;

% Training Loop
for epoch = 1:epochs
    total_loss = 0;
    grad_accum = zeros(size(grads));
    
    for i = 1:size(X, 2)
        x_val = X(:, i);
        y_val = Y(i);
        
        % Compute Loss
        [loss_val] = f_loss(x_val, y_val, W1_val, b1_val, W2_val, b2_val, W3_val, b3_val);
        total_loss = total_loss + loss_val;
        
        % Compute Gradients
        grad_vals = f_grad(x_val, y_val, W1_val, b1_val, W2_val, b2_val, W3_val, b3_val);
        
        % Update Parameters
        W1_val = full(W1_val - learning_rate * reshape(grad_vals(1:numel(W1)), size(W1)));
        b1_val = full(b1_val - learning_rate * reshape(grad_vals(numel(W1)+1:numel(W1)+numel(b1)), size(b1)));
        W2_val = full(W2_val - learning_rate * reshape(grad_vals(numel(W1)+numel(b1)+1:numel(W1)+numel(b1)+numel(W2)), size(W2)));
        b2_val = full(b2_val - learning_rate * reshape(grad_vals(numel(W1)+numel(b1)+numel(W2)+1:numel(W1)+numel(b1)+numel(W2)+numel(b2)), size(b2)));
        W3_val = full(W3_val - learning_rate * reshape(grad_vals(numel(W1)+numel(b1)+numel(W2)+numel(b2)+1:numel(W1)+numel(b1)+numel(W2)+numel(b2)+numel(W3)), size(W3)));
        b3_val = full(b3_val - learning_rate * reshape(grad_vals(numel(W1)+numel(b1)+numel(W2)+numel(b2)+numel(W3)+1:end), size(b3)));
    end
    full(total_loss)
    
end

% Test the Neural Network
x_test = -.1;
z1_test = W1_val * x_test + b1_val;
a1_test = relu(z1_test);
z2_test = W2_val * a1_test + b2_val;
a2_test = relu(z2_test);
z3_test = W3_val*a2_test + b3_val;
y_test = sigmoid(z3_test)
y_real = (x_test)^2 
