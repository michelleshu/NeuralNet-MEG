function [cost,grad] = getSAECost(theta, visibleSize, hiddenSize, ...
                                         lambda, sparsityParam, beta, data)

% visibleSize: the number of input units 
% hiddenSize: the number of hidden units 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units
% beta: weight of sparsity penalty term
% data: matrix containing the training data. data(:,i) is the i-th example. 

m = size(data, 2);  % number of training examples

% The input theta is a vector (because minFunc expects the parameters to be
% a vector). We first convert theta to the (W1, W2, b1, b2) format.

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), ...
     visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

% One initial pass to compute average activation of nodes in hidden layer
act2 = zeros(hiddenSize, 1);
for example = 1 : m
    a1 = data(:, example);
    z2 = W1 * a1 + b1;
    a2 = sigmoid(z2);
    act2 = act2 + a2;
end
act2 = act2 / m;
KL = sum(sparsityParam * log(sparsityParam ./ act2) + ...
    (1 - sparsityParam) * log((1 - sparsityParam) ./ (1 - act2)));
deltaKL = - (sparsityParam ./ act2) + ((1 - sparsityParam) ./ (1 - act2));

for example = 1 : m    
    % Forward propagate from input to hidden layer
    a1 = data(:, example);
    z2 = W1 * a1 + b1;
    a2 = sigmoid(z2);
    
    % Forward propagate from hidden to output layer
    z3 = W2 * a2 + b2;
    h = sigmoid(z3);
    
    % Compute costs
    cost = cost + ((h - a1)' * (h - a1) * 0.5);
    
    % Backpropagate output errors
    delta3 = (h - a1) .* h .* (1 - h); 
    
    % Backpropagate hidden layer error and add KL divergence term for 
    % sparsity penalty
    delta2 = ((W2' * delta3) + (beta * deltaKL)) .* (a2 .* (1 - a2));
    
    % Add result of this example to gradients.
    W1grad = W1grad + (delta2 * a1');
    W2grad = W2grad + (delta3 * a2');
    b1grad = b1grad + delta2;
    b2grad = b2grad + delta3;   
end

cost = cost / m;
% Regularize cost with weight decay term and add sparsity constraint
cost = cost + (lambda / 2 * (sum(sum(W1 .^2)) + sum(sum(W2 .^ 2)))) + ...
    (beta * KL);

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