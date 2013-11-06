% LEAVE TWO OUT CROSS-VALIDATION

% Select multiple target features by index
targetInds = [1:12, 16, 25, 36, 37, 45, 46, 53, 54, 55, 56, 72, 74, 75, ...
    76, 84, 86, 87, 91, 117, 121, 128, 144, 146, 154, 160, 169, 191, 195];


% Make random stream random
s = RandStream('mt19937ar','Seed','shuffle');
RandStream.setGlobalStream(s);

% Load split information
s = load('./splits.mat');
allSplits = s.splits;

% Parameters to specify
numComponents = 30;
numTimePoints = 30;
subject = 'A';
inputSize = numComponents * numTimePoints;
hiddenSize = 100;
outputSize = 40;
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

percentCorrect = zeros(5, 1);

for trial = 1 : 5
    fprintf('Trial %i\n', trial);
    
    % Track number of correct predictions
    numCorrect = 0;
    % Get splits for this trial
    splits = squeeze(allSplits(trial, :, :));

    for fold = 1 : 30
        testEx1 = splits(1, fold);
        testEx2 = splits(2, fold);
        % Make smaller number testEx1
        if testEx1 > testEx2
            tmp = testEx2;
            testEx2 = testEx1;
            testEx1 = tmp;
        end

        % Select two examples to leave out for test and train on rest
        fprintf('Test examples: %i and %i\n', testEx1, testEx2);

        train_inputs = vertcat(inputs(1 : testEx1 - 1, :), ...
            inputs(testEx1 + 1 : testEx2 - 1, :), ...
            inputs(testEx2 + 1 : end, :));
        train_targets = vertcat(targets(1 : testEx1 - 1, :), ...
            targets(testEx1 + 1 : testEx2 - 1, :), ...
            targets(testEx2 + 1 : end, :));

        test_input1 = inputs(testEx1, :)';
        test_target1 = targets(testEx1, :)'; 
        test_input2 = inputs(testEx2, :)';
        test_target2 = targets(testEx2, :)';

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
        test_pred1 = sigmoid(W2 * sigmoid(W1 * test_input1 + b1) + b2);
        test_pred2 = sigmoid(W2 * sigmoid(W1 * test_input2 + b1) + b2);
        
        d1 = pdist([test_pred1'; test_target1']) + ...
            pdist([test_pred2'; test_target2']);
        
        d2 = pdist([test_pred1'; test_target2']) + ...
            pdist([test_pred2'; test_target1']);

        if (d1 < d2)
            numCorrect = numCorrect + 1;
            fprintf('Correct: d1 = %1.5f; d2 = %1.2f\n', d1, d2);
        else
            fprintf('Incorrect: d1 = %1.5f; d2 = %1.2f\n\n', d1, d2);
        end

    end

    percentCorrect(trial) = numCorrect / 30;
    fprintf('Percent Correct: %2.3f\n %s\n', percentCorrect(trial), ...
        datestr(now));
end

save('percentCorrect2v2.mat', 'percentCorrect');