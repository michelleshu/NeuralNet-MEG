% LEAVE TWO OUT CROSS-VALIDATION

% Select multiple target features by index
%targetInds = [1:12, 16, 25, 36, 37, 45, 46, 53, 54, 55, 56, 72, 74, 75, ...
%    76, 84, 86, 87, 91, 117, 121, 128, 144, 146, 154, 160, 169, 191, 195];

% 50 features
targetInds = [173, 175, 202, 31, 206, 63, 64, 77, 115, 194, 197, 201, 47, ...
    53, 145, 187, 55, 71, 124, 128, 153, 190, 218, 37, 154, 169, 191, ...
    195, 36, 56, 117, 164, 16, 25, 46, 54, 74, 91, 121, 9, 45, 76, 84, ...
    86, 87, 75, 160, 72, 144, 146];

% 95 features
% targetInds = [28, 61, 27, 68, 113, 188, 193, 30, 41, 111, 165, 73, 166, ...
%     10, 69, 70, 147, 200, 44, 52, 148, 155, 176, 180, 196, 20, 29, 32, ...
%     80, 150, 192, 62, 65, 79, 162, 189, 33, 38, 157, 186, 209, 40, 42, ...
%     59, 66, 173, 175, 202, 31, 206, 63, 64, 77, 115, 194, 197, 201, 47, ...
%     53, 145, 187, 55, 71, 124, 128, 153, 190, 218, 37, 154, 169, 191, ...
%     195, 36, 56, 117, 164, 16, 25, 46, 54, 74, 91, 121, 9, 45, 76, 84, ...
%     86, 87, 75, 160, 72, 144, 146];

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
hiddenSize = 50;
outputSize = 50;
lambda = 1e-4;

% minFunc options
options.Method = 'lbfgs';
options.maxIter = 15000;
options.maxFunEvals = 15000;
options.TolX = 1e-6;
options.TolFun = 1e-6;
options.display = 'off';

% Get input and target data to use
inputs = getInputsFromPCA(subject, numComponents, numTimePoints);
targets = getTargets(targetInds, '../data/sem_matrix_bin.mat');

percentCorrect = zeros(5, 1);

for trial = 1
    fprintf('Trial %i\n', trial);
    
    % Track number of correct predictions
    numCorrect = 0;
    % Get splits for this trial
    splits = squeeze(allSplits(trial, :, :));

    for fold = 19 : 30
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

%         d1 = pdist([test_pred1'; test_target1'], 'cosine') + ...
%             pdist([test_pred2'; test_target2'], 'cosine');
%         
%         d2 = pdist([test_pred1'; test_target2'], 'cosine') + ...
%             pdist([test_pred2'; test_target1'], 'cosine');

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