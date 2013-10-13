%% trainStackedSAE.m
% Train a sparse autoencoder on MEG data, based on Stanford UFLDL code
% Michelle Shu | October 11, 2013

addpath classifier
addpath minFunc

subject = 'A';
resultsDir = './stacked_results';
dataDir = './data';
accDir = './stacked_results/accuracy';
semMatrix = './data/sem_matrix.mat';

% -------------------------------------------------------------------------
% Parameters:
visibleSize = 306;      % number of input units
hiddenSize1 = 100;      % number of hidden units in first hidden layer
hiddenSize2 = 100;      % number of hidden units in second hidden layer
sparsityParam = 0.1;    % desired average activation of hidden units
lambda = 0.0001;        % weight decay parameter
beta = 3;               % weight of sparsity penalty

dataFile = sprintf('%s/%s_raw_avrg.mat', dataDir, subject);
networkWeightsFile = sprintf('%s/network/%s_network.mat', ...
    resultsDir, subject);
sparse1File = sprintf('%s/sparse1/%s_sparse1.mat', resultsDir, subject);
sparse2File = sprintf('%s/sparse2/%s_sparse2.mat', resultsDir, subject);
classify1Dir = sprintf('%s/classify1', resultsDir);  % classifier results 
classify2Dir = sprintf('%s/classify2', resultsDir);
acc1File = sprintf('%s/%s_acc1.mat', accDir, subject); % layer 1 mean acc
acc2File = sprintf('%s/%s_acc2.mat', accDir, subject); % layer 2 mean acc

% -------------------------------------------------------------------------
% Initialization 1:
% Get patches, time series, labels from raw input
[patches, time, words] = getData(dataFile);

% Obtain random initialization for weight parameters
theta = initializeParameters(hiddenSize1, visibleSize);

% -------------------------------------------------------------------------
% Verify correctness of cost function by checking results against numerical
% gradient result
% [~, grad] = getSAECost(theta, visibleSize, hiddenSize, lambda, ...
%                                      sparsityParam, beta, patches);
% numgrad = computeNumericalGradient( @(x) getSAECost(x, visibleSize, ...
%                                                  hiddenSize, lambda, ...
%                                                  sparsityParam, beta, ...
%                                                  patches), theta);
% disp([numgrad grad]); 
% % Compare numerically computed gradients with the ones from backprop
% diff = norm(numgrad-grad)/norm(numgrad+grad);
% disp(diff); % Should be small, less than 1e-9.

% -------------------------------------------------------------------------
% Train sparse autoencoder with minFunc (L-BFGS) library
% addpath code/minFunc/
options.Method = 'lbfgs';
options.maxIter = 10;
options.maxFunEvals = 10;
options.display = 'on';
 
[opttheta, cost] = minFunc( @(p) getSAECost(p, ...
                                   visibleSize, hiddenSize1, ...
                                   lambda, sparsityParam, ...
                                   beta, patches), ...
                                   theta, options);

W1 = reshape(opttheta(1 : hiddenSize1 * visibleSize), hiddenSize1, ...
    visibleSize);
W4 = reshape(theta(hiddenSize1 * visibleSize + 1 : 2 * hiddenSize1 * ...
    visibleSize), visibleSize, hiddenSize1);
b1 = opttheta(2 * hiddenSize1 * visibleSize + 1 : 2 * hiddenSize1 * ...
    visibleSize + hiddenSize1);
b4 = theta(2 * hiddenSize1 * visibleSize + hiddenSize1 + 1 : end);

% -------------------------------------------------------------------------
% Run all MEG images through first hidden layer of SAE to get sparse (1)
% representation
numWords = numel(words);
sparseDim = size(W1, 1); % sparse vector length
patchesPerWord = size(patches, 2) / numWords;

data = zeros(numWords, sparseDim, patchesPerWord);

for word = 1 : numWords
    for patch = 1 : patchesPerWord
        i = (word - 1) * patchesPerWord + patch;  % index of current patch
        % Forward propagate from input to hidden layer
        a1 = patches(:, i);
        z2 = W1 * a1 + b1;
        a2 = sigmoid(z2);
        data(word, :, patch) = a2;
    end
end

save(sparse1File, 'data', 'time', 'words');

% -------------------------------------------------------------------------
% Use sparse vector representations as inputs to classifier
% addpath classifier/
%classifyMagFeatsStacked(subject, sparse1File, classify1Dir, semMatrix);

% -------------------------------------------------------------------------
% Compute classification accuracy for first layer only
%acc_ones = zeros(5, 1);  % 1 v 2 acc over 5 trials
%acc_twos = zeros(5, 1);  % 2 v 2 acc over 5 trials
%for trial = 1 : 5
%    classifyFile = sprintf('%s/%s/%s_sparse_%i.mat', classify1Dir, ...
%        subject, subject, trial);
%    [acc_ones(trial), acc_twos(trial)] = getAccuracy(classifyFile);
%end
%fprintf('1 v 2 Accuracy: %2.3f\n', mean(acc_ones));
%fprintf('2 v 2 Accuracy: %2.3f\n', mean(acc_twos));

%save(acc1File, 'acc_ones', 'acc_twos');


%% Start 2nd hidden layer -------------------------------------------------

% Get patches, time series, labels from sparse representations from first
% hidden layer
[patches, time, words] = getData(sparse1File);

% Obtain random initialization for weight parameters
theta = initializeParameters(hiddenSize2, hiddenSize1);

[opttheta, cost] = minFunc( @(p) getSAECost(p, ...
                                   hiddenSize1, hiddenSize2, ...
                                   lambda, sparsityParam, ...
                                   beta, patches), ...
                                   theta, options);

W2 = reshape(opttheta(1 : hiddenSize2 * hiddenSize1), hiddenSize2, ...
    hiddenSize1);
W3 = reshape(theta(hiddenSize2 * hiddenSize1 + 1 : 2 * hiddenSize2 * ...
    hiddenSize1), hiddenSize1, hiddenSize2);
b2 = opttheta(2 * hiddenSize2 * hiddenSize1 + 1 : 2 * hiddenSize2 * ...
    hiddenSize1 + hiddenSize2);
b3 = theta(2 * hiddenSize2 * hiddenSize1 + hiddenSize2 + 1 : end);

% Save network params
save(networkWeightsFile, 'W1', 'W2', 'W3', 'W4', 'b1', 'b2', 'b3', 'b4');

% -------------------------------------------------------------------------
% Run all SAE layer 1 representations through 2nd hidden layer to get 
% sparse (2) representation
numWords = numel(words);
sparseDim = size(W2, 1); % sparse vector length
patchesPerWord = size(patches, 2) / numWords;

data = zeros(numWords, sparseDim, patchesPerWord);

for word = 1 : numWords
    for patch = 1 : patchesPerWord
        i = (word - 1) * patchesPerWord + patch;  % index of current patch
        % Forward propagate from input to hidden layer
        a1 = patches(:, i);
        z2 = W2 * a1 + b1;
        a2 = sigmoid(z2);
        data(word, :, patch) = a2;
    end
end

save(sparse2File, 'data', 'time', 'words');

% -------------------------------------------------------------------------
% Second layer classifier results
classifyMagFeatsStacked(subject, sparse2File, classify2Dir, semMatrix);

acc_ones = zeros(5, 1);  % 1 v 2 acc over 5 trials
acc_twos = zeros(5, 1);  % 2 v 2 acc over 5 trials
for trial = 1 : 5
    classifyFile = sprintf('%s/%s/%s_sparse_%i.mat', classify2Dir, ...
        subject, subject, trial);
    [acc_ones(trial), acc_twos(trial)] = getAccuracy(classifyFile);
end
fprintf('1 v 2 Accuracy: %2.3f\n', mean(acc_ones));
fprintf('2 v 2 Accuracy: %2.3f\n', mean(acc_twos));

save(acc2File, 'acc_ones', 'acc_twos');
