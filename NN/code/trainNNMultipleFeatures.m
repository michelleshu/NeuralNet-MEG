% LEAVE ONE OUT CROSS-VALIDATION

% Select multiple target features by index
targetInds = [1:12, 16, 45, 53, 55, 56, 72, 75, 76, 84, 86, 87, 121, ...
    128, 144, 146, 154, 160, 169];

% Make random stream random
s = RandStream('mt19937ar','Seed','shuffle');
RandStream.setGlobalStream(s);

% Parameters to specify
numComponents = 30;
numTimePoints = 30;
subject = 'A';
inputSize = numComponents * numTimePoints;
hiddenSize = 20;
outputSize = 30;
lambda = 1e-4;

% minFunc options
options.Method = 'lbfgs';
options.maxIter = 10000;
options.maxFunEvals = 10000;
options.TolX = 1e-6;
options.TolFun = 1e-6;
options.display = 'off';

% Get input and target data to use
inputs = getInputsFromPCA(subject, numComponents, numTimePoints);
targets = getTargets(targetInds, '../data/sem_matrix_bin.mat');

errorNorms = zeros(60, 1);
percentCorrect = zeros(60, 1);

for testEx = 1 : 10 : 60
% Select one example to leave out for test and train on rest
    fprintf('Test example: %i\n', testEx);

    train_inputs = vertcat(inputs(1 : testEx - 1, :), ...
        inputs(testEx + 1 : end, :));
    train_targets = vertcat(targets(1 : testEx - 1, :), ...
        targets(testEx + 1 : end, :));

    test_input = inputs(testEx, :)';
    test_target = targets(testEx, :)'; 

    theta = rand((inputSize + 1) * hiddenSize + ...
        (hiddenSize + 1) * outputSize, 1);

    % Train network on training examples

    [opttheta, cost] = minFunc( @(p) getNNCost(p, inputSize, ...
                                hiddenSize, outputSize, lambda, ...
                                train_inputs, train_targets), ...
                                theta, options);

    W1 = reshape(opttheta(1 : hiddenSize * inputSize), hiddenSize, ...
         inputSize);
    W2 = reshape(opttheta(hiddenSize * inputSize + 1 : ...
         hiddenSize * (inputSize + outputSize)), outputSize, hiddenSize);
    b1 = opttheta(hiddenSize * (inputSize + outputSize) + 1 : ...
         hiddenSize * (inputSize + outputSize + 1));
    b2 = opttheta(hiddenSize * (inputSize + outputSize + 1) + 1 : end); 

    % Get prediction for test example
    test_pred = sigmoid(W2 * sigmoid(W1 * test_input + b1) + b2);

    errorNorms(testEx) = norm(test_target - test_pred);
    
    correct = 0;
    for feat = 1 : numel(test_pred) 
        if (abs(test_pred(feat) - test_target(feat) < 0.5))
            correct = correct + 1;
        end
    end
    percentCorrect(testEx) = correct / numel(test_pred);
    disp([testEx percentCorrect(testEx)]);
end
    