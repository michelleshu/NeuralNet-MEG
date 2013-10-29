%% trainSAE.m
% Train a sparse autoencoder on MEG data, based on Stanford UFLDL code
% Michelle Shu | September 17, 2013

function trainSAE(sparsityParam, hiddenSize, subject, resultsDir, ...
    dataDir, semMatrix)

% -------------------------------------------------------------------------
% Parameters:
visibleSize = 306;      % number of input units
%hiddenSize = 100;       % number of hidden units
%sparsityParam = 0.1;    % desired average activation of hidden units
lambda = 0.0001;        % weight decay parameter
beta = 3;               % weight of sparsity penalty

dataFile = sprintf('%s/%s_raw_avrg.mat', dataDir, subject);
networkWeightsFile = sprintf('%s/network/%s_%1.3fR_%iK.mat', ...
    resultsDir, subject, sparsityParam, hiddenSize);
sparseRepFile = sprintf('%s/sparse/%s_%1.3fR_%iK.mat', resultsDir, ...
    subject, sparsityParam, hiddenSize);
classifyDir = sprintf('%s/classify', resultsDir);  % classifier results            

% -------------------------------------------------------------------------
% Initialization:
% Get patches, time series, labels from raw input
[patches, time, words] = getData(dataFile);

% Obtain random initialization for weight parameters
theta = initializeParameters(hiddenSize, visibleSize);

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
options.maxIter = 3000;
options.maxFunEvals = 3000;
options.display = 'on';
 
[opttheta, cost] = minFunc( @(p) getSAECost(p, ...
                                   visibleSize, hiddenSize, ...
                                   lambda, sparsityParam, ...
                                   beta, patches), ...
                                   theta, options);

W1 = reshape(opttheta(1 : hiddenSize * visibleSize), hiddenSize, ...
    visibleSize);
W2 = reshape(opttheta(hiddenSize * visibleSize + 1 : 2 * hiddenSize * ...
    visibleSize), visibleSize, hiddenSize);
b1 = opttheta(2 * hiddenSize * visibleSize + 1 : 2 * hiddenSize * ...
    visibleSize + hiddenSize);
b2 = opttheta(2 * hiddenSize * visibleSize + hiddenSize + 1 : end);

save(networkWeightsFile, 'W1', 'W2', 'b1', 'b2');

% -------------------------------------------------------------------------
% Run all MEG images through hidden layer of SAE to get sparse 
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

save(sparseRepFile, 'data', 'time', 'words');

% -------------------------------------------------------------------------
% Use sparse vector representations as inputs to classifier
% addpath classifier/
classifyMagFeats(subject, sparseRepFile, classifyDir, sparsityParam, ...
    hiddenSize, semMatrix);

% -------------------------------------------------------------------------
% Compute classification accuracy
%acc_ones = zeros(5, 1);  % 1 v 2 acc over 5 trials
%acc_twos = zeros(5, 1);  % 2 v 2 acc over 5 trials
%for trial = 1 : 5
%    classifyFile = sprintf('%s/%s/%s_sparse_%1.3fR_%iK_%i.mat', classifyDir, ...
%        subject, subject, sparsityParam, hiddenSize, trial);
%    [acc_ones(trial), acc_twos(trial)] = getAccuracy(classifyFile);
%end
%fprintf('1 v 2 Accuracy: %2.3f\n', mean(acc_ones));
%fprintf('2 v 2 Accuracy: %2.3f\n', mean(acc_twos));

