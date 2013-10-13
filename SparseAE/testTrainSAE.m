%% testTrainSAE.m
% Train a sparse autoencoder on MEG data, based on Stanford UFLDL code
% Michelle Shu | September 17, 2013
% -------------------------------------------------------------------------
% Parameters:
visibleSize = 3;      % number of input units
hiddenSize = 1;       % number of hidden units
sparsityParam = 0.6251;    % desired average activation of hidden units
lambda = 0;        % weight decay parameter
beta = 0;               % weight of sparsity penalty           

% ------------------------------------------------------------------------

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
options.maxIter = 500;
options.maxFunEvals = 500;
options.display = 'on';
options.TolX = 1e-15;
options.TolFun = 1e-15;
 
[opttheta, cost] = minFunc( @(p) testGetSAECost(p, ...
                                   visibleSize, hiddenSize, ...
                                   lambda, sparsityParam, ...
                                   beta, examples_in, examples_out), ...
                                   theta, options);

W1 = reshape(opttheta(1 : hiddenSize * visibleSize), hiddenSize, ...
    visibleSize);
W2 = reshape(opttheta(hiddenSize * visibleSize + 1 : 2 * hiddenSize * ...
    visibleSize), visibleSize, hiddenSize);
b1 = opttheta(2 * hiddenSize * visibleSize + 1 : 2 * hiddenSize * ...
    visibleSize + hiddenSize);
b2 = opttheta(2 * hiddenSize * visibleSize + hiddenSize + 1 : end);

