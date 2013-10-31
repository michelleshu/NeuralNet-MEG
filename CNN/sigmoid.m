function sigm = sigmoid(x)
%% Simple sigmoid activation function
    sigm = 1 ./ (1 + exp(-x));
end