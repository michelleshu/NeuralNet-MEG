%% trainSAE.m
% Train a sparse autoencoder on MEG data, based on Stanford UFLDL code
% Michelle Shu | September 17, 2013
function trainSAE(sparsityParam)

% -------------------------------------------------------------------------
% Parameters:
visibleSize = 306;      % number of input units
hiddenSize = 306;       % number of hidden units
%sparsityParam = 0.01;   % desired average activation of hidden units
lambda = 0.0001;        % weight decay parameter
beta = 3;               % weight of sparsity penalty

subject = 'D';
dataFile = 'data/D_raw_avrg.mat';           % file to load MEG data from
semMatrixFile = 'data/sem_matrix.mat';      % file with semantic features
sparseRepFile = 'results/sparseRepresentation/D_sparse.mat';  
                                            % save sparse vectors here
classifyDir = 'results/classifierResults';  % classifier results            

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
addpath minFunc/
options.Method = 'lbfgs';
options.maxIter = 3000;
options.maxFunEvals = 3500;
options.display = 'on';
 
[opttheta, cost] = minFunc( @(p) getSAECost(p, ...
                                   visibleSize, hiddenSize, ...
                                   lambda, sparsityParam, ...
                                   beta, patches), ...
                                   theta, options);

W1 = reshape(opttheta(1 : hiddenSize * visibleSize), hiddenSize, ...
    visibleSize);
b1 = opttheta(2 * hiddenSize * visibleSize + 1 : 2 * hiddenSize * ...
    visibleSize + hiddenSize);

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
addpath classifier/
load(semMatrixFile);
classifyMagFeats(subject, sparseRepFile, classifyDir, sem_matrix);

% -------------------------------------------------------------------------
% Compute classification accuracy
acc_ones = zeros(5, 1);  % 1 v 2 acc over 5 trials
acc_twos = zeros(5, 1);  % 2 v 2 acc over 5 trials
for trial = 1 : 5
    classifyFile = sprintf('%s/%s/%s_sparse_%i.mat', classifyDir, ...
        subject, subject, trial);
    [acc_ones(trial), acc_twos(trial)] = getAccuracy(classifyFile);
end
fprintf('1 v 2 Accuracy: %2.3f\n', mean(acc_ones));
fprintf('2 v 2 Accuracy: %2.3f\n', mean(acc_twos));

end