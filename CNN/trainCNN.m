% Make random stream random
s = RandStream('mt19937ar','Seed','shuffle');
RandStream.setGlobalStream(s);

% Parameters to specify
subject = 'A';
inputSliceSize = 306;
timeSize = 15;
hidden1SliceSize = 50;
hidden2Size = 5;
outputSize = 1;
lambda = 0;

% minFunc options
options.Method = 'lbfgs';
options.maxIter = 1000;
options.maxFunEvals = 1000;
options.TolX = 1e-9;
options.display = 'off';

percentCorrect = zeros(218, 1);

for targetFeature = 1 : 50
    fprintf('Target feature: %i\n', targetFeature);

    % Get input and target data to use
    inputs = getInputs(subject, timeSize);
    targets = getTargets(targetFeature, '../data/sem_matrix_bin.mat');

    % Track number of correct predictions
    numCorrect = 0;

    for testEx = 1 : size(targets)
    % Select one example to leave out for test and train on rest
        fprintf('Test example: %i\n', testEx);

        train_inputs = cat(1, inputs(1 : testEx - 1, :, :), ...
            inputs(testEx + 1 : end, :, :));
        train_targets = vertcat(targets(1 : testEx - 1), ...
            targets(testEx + 1 : end));

        test_input = squeeze(inputs(testEx, :, :));
        test_target = targets(testEx); 

        theta = rand(hidden1SliceSize * (inputSliceSize + timeSize * ...
            hidden2Size + timeSize) + hidden2Size * (outputSize + 1) + ...
            outputSize, 1);

        % Train network on training examples

        [opttheta, cost] = minFunc( @(p) getCNNCost(p, inputSliceSize, ...
                                    timeSize, hidden1SliceSize, ...
                                    hidden2Size, outputSize, lambda, ...
                                    train_inputs, train_targets), ...
                                    theta, options);

        W1 = reshape(opttheta(1 : inputSliceSize * hidden1SliceSize), ...
             hidden1SliceSize, inputSliceSize);
        i = inputSliceSize * hidden1SliceSize; 

        W2 = reshape(opttheta(i + 1 : i + hidden1SliceSize * timeSize * ...
            hidden2Size), hidden2Size, hidden1SliceSize * timeSize);
        i = i + hidden1SliceSize * timeSize * hidden2Size;

        W3 = reshape(opttheta(i + 1 : i + hidden2Size * outputSize), ...
            outputSize, hidden2Size);
        i = i + hidden2Size * outputSize;

        b1 = reshape(opttheta(i + 1 : i + hidden1SliceSize * timeSize), ...
            hidden1SliceSize, timeSize);
        i = i + hidden1SliceSize * timeSize;
        b2 = opttheta(i + 1 : i + hidden2Size);
        i = i + hidden2Size;
        b3 = opttheta(i + 1 : i + outputSize);

        % Get prediction for test example
        a2 = sigmoid(W1 * test_input + b1);
        a2 = a2(:);
        a3 = sigmoid(W2 * a2 + b2);
        test_pred = sigmoid(W3 * a3 + b3);
        

        fprintf('Target: %1.3f\n', test_target);
        fprintf('Prediction: %1.3f\n\n', test_pred);

        if (test_pred > 0.5)
            test_pred = 1;
        else
            test_pred = 0;
        end

        if (test_pred == test_target)
            numCorrect = numCorrect + 1;
        end

    end

    percentCorrect(targetFeature) = numCorrect / 60;
    fprintf('Percent Correct: %2.3f\n', percentCorrect(targetFeature));
    
end