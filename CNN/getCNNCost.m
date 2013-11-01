function [cost,grad] = getCNNCost(theta, inputSliceSize, timeSize, ...
    hidden1SliceSize, hidden2Size, outputSize, lambda, inputs, targets)

% inputSliceSize: dimensions of one "snapshot"/"slice" (i.e. # of sensors)
% timeSize: number of time points (after averaging, compression)
% hidden1SliceSize: first hidden layer hidden units per slice
% hidden2Size: total # units in second hidden layer
% outputSize: number of output units

% lambda: weight decay parameter
% inputs: training input data (m x inputSliceSize x timeSize))
% targets: training target data (m x 1 for one target)

m = size(inputs, 1);  % number of training examples

% The input theta is a vector (because minFunc expects the parameters to be
% a vector). We first convert theta to the (W1, W2, W3, b1, b2, b3) format.

W1 = reshape(theta(1 : inputSliceSize * hidden1SliceSize), ...
     hidden1SliceSize, inputSliceSize);
i = inputSliceSize * hidden1SliceSize; % track where we are in theta

W2 = reshape(theta(i + 1 : i + hidden1SliceSize * timeSize * hidden2Size), ...
     hidden2Size, hidden1SliceSize * timeSize);
i = i + hidden1SliceSize * timeSize * hidden2Size;

W3 = reshape(theta(i + 1 : i + hidden2Size * outputSize), outputSize, ...
    hidden2Size);
i = i + hidden2Size * outputSize;

b1 = reshape(theta(i + 1 : i + hidden1SliceSize * timeSize), ...
    hidden1SliceSize, timeSize);
i = i + hidden1SliceSize * timeSize;
b2 = theta(i + 1 : i + hidden2Size);
i = i + hidden2Size;
b3 = theta(i + 1 : i + outputSize);


% Cost and gradient variables
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
W3grad = zeros(size(W3));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));
b3grad = zeros(size(b3));

for ex = 1 : m  
    
    input = squeeze(inputs(ex, :, :));
    target = targets(ex);
    
    % Column-wise forward propagation of each vector from input to hidden1
    a2 = sigmoid(W1 * input + b1);
    
    % Roll out into one vector
    a2 = a2(:);
    
    % Forward propagate from hidden1 to hidden2
    a3 = sigmoid(W2 * a2 + b2);
    
    % Forward propagate from hidden2 to output h
    h = sigmoid(W3 * a3 + b3);
    
    % Compute costs
    cost = cost + ((h - target)' * (h - target) * 0.5);
    
    % Backpropagate output errors
    delta4 = (h - target) .* h .* (1 - h); 
    
    % Backpropagate hidden2 errors
    delta3 = (W3' * delta4) .* a3 .* (1 - a3);
    
    % Backpropagate hidden1 errors
    delta2 = (W2' * delta3) .* a2 .* (1 - a2);
    
    % Add result of this example to gradients.
    % For the first layer, the gradient averages result from each slice
    delta2 = reshape(delta2, hidden1SliceSize, timeSize);
    W1grad = W1grad + delta2 * input';
    %W1grad = W1grad + (delta2 * input' / timeSize);
    W2grad = W2grad + (delta3 * a2');
    W3grad = W3grad + (delta4 * a3');
    b1grad = b1grad + delta2;
    b2grad = b2grad + delta3;
    b3grad = b3grad + delta4;
end

cost = cost / m;
% Regularize cost with weight decay term and add sparsity constraint
cost = cost + (lambda / 2 * (sum(sum(W1 .^2)) + sum(sum(W2 .^ 2)) + ...
    sum(sum(W3 .^ 2))));

% Regularize gradients for non-bias weights with weight decay terms
W1grad = W1grad / m + lambda * W1;
W2grad = W2grad / m + lambda * W2;
W3grad = W3grad / m + lambda * W3;
b1grad = b1grad / m;
b2grad = b2grad / m;
b3grad = b3grad / m;

%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).

grad = [W1grad(:) ; W2grad(:) ; W3grad(:) ; b1grad(:) ; b2grad(:) ; ...
    b3grad(:)];

end