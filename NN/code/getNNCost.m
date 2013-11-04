function [cost,grad] = getNNCost(theta, inputSize, hiddenSize, ...
                                 outputSize, lambda, inputs, targets)

% inputSize: the number of input units 
% hiddenSize: the number of hidden units 
% outputSize: the number of output units (targets)
% lambda: weight decay parameter
% inputs: training input data
% targets: training target data

m = size(inputs, 1);  % number of training examples

% The input theta is a vector (because minFunc expects the parameters to be
% a vector). We first convert theta to the (W1, W2, b1, b2) format.

W1 = reshape(theta(1 : hiddenSize * inputSize), hiddenSize, inputSize);
W2 = reshape(theta(hiddenSize * inputSize + 1 : ...
     hiddenSize * (inputSize + outputSize)), outputSize, hiddenSize);
b1 = theta(hiddenSize * (inputSize + outputSize) + 1 : ...
     hiddenSize * (inputSize + outputSize + 1));
b2 = theta(hiddenSize * (inputSize + outputSize + 1) + 1 : end);

% Cost and gradient variables
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

for example = 1 : m    
    % Get input and target
    a1 = inputs(example, :)';
    t = targets(example, :)';
    
    % Forward propagate from input to hidden layer
    z2 = W1 * a1 + b1;
    a2 = sigmoid(z2);
    
    % Forward propagate from hidden to output layer
    z3 = W2 * a2 + b2;
    h = sigmoid(z3);
    
    % Compute costs
    cost = cost + ((h - t)' * (h - t) * 0.5);
    
    % Backpropagate output errors
    delta3 = (h - t) .* h .* (1 - h); 
    
    % Backpropagate hidden layer error
    delta2 = (W2' * delta3) .* (a2 .* (1 - a2));
    
    % Add result of this example to gradients.
    W1grad = W1grad + (delta2 * a1');
    W2grad = W2grad + (delta3 * a2');
    b1grad = b1grad + delta2;
    b2grad = b2grad + delta3;   
end

cost = cost / m;
% Regularize cost with weight decay term and add sparsity constraint
cost = cost + (lambda / 2 * (sum(sum(W1 .^2)) + sum(sum(W2 .^ 2))));

% Regularize gradients for non-bias weights with weight decay terms
W1grad = W1grad / m + lambda * W1;
W2grad = W2grad / m + lambda * W2;
b1grad = b1grad / m;
b2grad = b2grad / m;

%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end