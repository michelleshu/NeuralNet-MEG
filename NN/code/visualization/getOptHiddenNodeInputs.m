function optInputs = getOptHiddenNodeInputs( W1 )
% Gets inputs vectors that will each optimally activate one of the hidden
% units in the network

hiddenSize = size(W1, 1);
inputSize = size(W1, 2);
optInputs = zeros(inputSize, hiddenSize);

for h = 1 : hiddenSize
    hiddenWeights = squeeze(W1(h, :));
    optInputs(:, h) = hiddenWeights ./ norm(hiddenWeights);
end

