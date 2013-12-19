% Select multiple target features by index
%targetInds = [1:12, 16, 25, 36, 37, 45, 46, 53, 54, 55, 56, 72, 74, 75, ...
%    76, 84, 86, 87, 91, 117, 121, 128, 144, 146, 154, 160, 169, 191, 195];
targetInds = 10;

% Make random stream random
s = RandStream('mt19937ar','Seed','shuffle');
RandStream.setGlobalStream(s);

% Parameters to specify
numComponents = 30;
numTimePoints = 30;
subject = 'A';
inputSize = numComponents * numTimePoints;
hiddenSize = 100;
outputSize = 1;
lambda = 1e-4;

% minFunc options
options.Method = 'lbfgs';
options.maxIter = 15000;
options.maxFunEvals = 15000;
options.TolX = 1e-10;
options.TolFun = 1e-10;
options.display = 'on';

% Get input and target data to use
inputs = getInputsFromPCA(subject, numComponents, numTimePoints);
targets = getTargets(targetInds, '../data/sem_matrix_bin.mat');

theta = rand((inputSize + 1) * hiddenSize + ...
    (hiddenSize + 1) * outputSize, 1);

% Train network on training examples
[opttheta, cost] = minFunc( @(p) getNNCost(p, inputSize, ...
                            hiddenSize, outputSize, lambda, ...
                            inputs, targets), ...
                            theta, options);

W1 = reshape(opttheta(1 : hiddenSize * inputSize), hiddenSize, ...
     inputSize);
W2 = reshape(opttheta(hiddenSize * inputSize + 1 : ...
     hiddenSize * (inputSize + outputSize)), outputSize, hiddenSize);
b1 = opttheta(hiddenSize * (inputSize + outputSize) + 1 : ...
     hiddenSize * (inputSize + outputSize + 1));
b2 = opttheta(hiddenSize * (inputSize + outputSize + 1) + 1 : end); 

save(sprintf('./code/visualization/weights/%s_one_feat_weights_z.mat', subject), ...
    'W1', 'W2', 'b1', 'b2');
    