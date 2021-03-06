% LEAVE ONE OUT CROSS-VALIDATION

% Make random stream random
s = RandStream('mt19937ar','Seed','shuffle');
RandStream.setGlobalStream(s);

% Parameters to specify
numComponents = 50;
numTimePoints = 30;
subject = 'A';
inputSize = numComponents * numTimePoints;
hiddenSize = 2;
outputSize = 1;
lambda = 1e-4;

% minFunc options
options.Method = 'lbfgs';
options.maxIter = 5000;
options.maxFunEvals = 5000;
options.TolX = 1e-10;
options.TolFun = 1e-10;
options.display = 'off';

percentCorrect = zeros(218, 1);

for targetFeature = 86
    fprintf('Target feature: %i\n', targetFeature);

    % Get input and target data to use
    inputs = getInputsFromPCAUA(subject, numComponents, numTimePoints);
    targets = getTargetsUA(subject, targetFeature, '../data/sem_matrix_bin.mat');

    % Track number of correct predictions
    numCorrect = 0;

    for testEx = 1 : size(targets, 1)
    % Select one example to leave out for test and train on rest
        fprintf('Test example: %i\n', testEx);

        train_inputs = vertcat(inputs(1 : testEx - 1, :), ...
            inputs(testEx + 1 : end, :));
        train_targets = vertcat(targets(1 : testEx - 1), ...
            targets(testEx + 1 : end));

        test_input = inputs(testEx, :)';
        test_target = targets(testEx); 

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

        fprintf('Target: %1.3f\n', test_target);
        fprintf('Prediction: %1.3f\n\n', test_pred);

        if (test_pred > 0.5)
            test_pred = 1;
        else
            test_pred = 0;
        end

        if (test_pred == test_target)
            numCorrect = numCorrect + 1;
            disp('Correct');
        else
            disp('Incorrect');
        end
    end

    percentCorrect(targetFeature) = numCorrect / size(targets, 1);
    fprintf('Percent Correct: %2.3f\n', percentCorrect(targetFeature));
    
end