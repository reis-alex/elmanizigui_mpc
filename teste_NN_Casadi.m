% Neural Network Training with CasADi in MATLAB
clc;
clear;

import casadi.*

% Define Hyperparameters
input_dim = 1;       % Number of input features
hidden_dim = 4;      % Number of hidden neurons
output_dim = 1;      % Number of output neurons
learning_rate = 0.5; % Learning rate
epochs = 1000;       % Number of epochs

% Generate Toy Dataset
rng(42); % For reproducibility
X = rand(input_dim, 100); % 100 samples, 2 features
Y = sin(X(1,:));%sum(X, 1) > 1;        % Output is 1 if sum of inputs > 1
% Y = double(Y);            % Convert logical to double

% Symbolic Variables
x = MX.sym('x', input_dim);     % Input
y_true = MX.sym('y_true', 1);   % True output

% Initialize Weights and Biases
W1 = MX.sym('W1', hidden_dim, input_dim); % Input to hidden weights
b1 = MX.sym('b1', hidden_dim, 1);         % Hidden layer biases
W2 = MX.sym('W2', output_dim, hidden_dim); % Hidden to output weights
b2 = MX.sym('b2', output_dim, 1);         % Output layer biases

% Define Neural Network
relu = @(z) max(0, z); % ReLU activation
sigmoid = @(z) 1 ./ (1 + exp(-z)); % Sigmoid activation

% Forward Pass
z1 = W1 * x + b1;
a1 = relu(z1);
z2 = W2 * a1 + b2;
y_pred = sigmoid(z2);

% Loss Function (Binary Cross-Entropy)
% loss = -y_true * log(y_pred) - (1 - y_true) * log(1 - y_pred);
loss = (y_true-y_pred).^2;
% Gradients Using CasADi
params = {W1, b1, W2, b2}; % Parameters to optimize
grads = gradient(loss, [W1(:); b1(:); W2(:); b2(:)]);

% CasADi Function for Forward and Backward Pass
f_loss = Function('f_loss', {x, y_true, W1, b1, W2, b2}, {loss});
f_grad = Function('f_grad', {x, y_true, W1, b1, W2, b2}, {grads});

% Initialize Parameters
W1_val = randn(hidden_dim, input_dim) * 0.1;
b1_val = randn(hidden_dim, 1) * 0.1;
W2_val = randn(output_dim, hidden_dim) * 0.1;
b2_val = randn(output_dim, 1) * 0.1;

% Training Loop
for epoch = 1:epochs
    total_loss = 0;
    grad_accum = zeros(size(grads));
    
    for i = 1:size(X, 2)
        x_val = X(:, i);
        y_val = Y(i);
        
        % Compute Loss
        [loss_val] = f_loss(x_val, y_val, W1_val, b1_val, W2_val, b2_val);
        total_loss = total_loss + loss_val;
        
        % Compute Gradients
        grad_vals = f_grad(x_val, y_val, W1_val, b1_val, W2_val, b2_val);
        
        % Update Parameters
        W1_val = full(W1_val - learning_rate * reshape(grad_vals(1:numel(W1)), size(W1)));
        b1_val = full(b1_val - learning_rate * reshape(grad_vals(numel(W1)+1:numel(W1)+numel(b1)), size(b1)));
        W2_val = full(W2_val - learning_rate * reshape(grad_vals(numel(W1)+numel(b1)+1:numel(W1)+numel(b1)+numel(W2)), size(W2)));
        b2_val = full(b2_val - learning_rate * reshape(grad_vals(end-numel(b2)+1:end), size(b2)));
    end
    
    % Display Loss Every 100 Epochs
    if mod(epoch, 100) == 0
        fprintf('Epoch %d: Loss = %.4f\n', epoch, full(total_loss / size(X, 2)));
    end
end

% Test the Neural Network
x_test = [0.4];
z1_test = W1_val * x_test + b1_val;
a1_test = relu(z1_test);
z2_test = W2_val * a1_test + b2_val;
y_test = sigmoid(z2_test)
y_real = sin(x_test(1))
% fprintf('Test Input: [%.2f, %.2f]\n', full(x_test(1)), full(x_test(2)));
% fprintf('Predicted Output: %.4f\n', full(y_test));