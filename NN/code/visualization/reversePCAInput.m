function origInputs = reversePCAInput(subject, inputVectors, ...
    numComponents, timeSize)
% Takes input as vectors, reshapes them, return as original input (sensors
% x time course)

% Retrieve PCA coefficient matrix
load(sprintf('../data/pca/%s_coeff.mat', subject));
coeff = coeff(:, 1 : numComponents);

origInputs = zeros(size(inputVectors, 2), size(coeff, 1), timeSize);

for i = 1 : size(inputVectors, 2)
    input = reshape(inputVectors(:, i), numComponents, timeSize);
    input = coeff * input;
    origInputs(i, :, :) = input;
end
