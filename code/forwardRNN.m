function forwardRNN(filters, params)
% Take final output from CNN and stack an RNN on top.
% At each level, children will be pooled according to progressively more 
% general brain regions.

% Extract features from raw data
data = extractFeaturesAllWords(filters, params);
time = params.time;
words = params.wordNames;

rnn = initRandomRNNWeights(params);

% Forward prop training data
disp('Forward Propagating through RNNs...');
% Output: numTrain x numRNN x numHid
data = forward(data, rnn, params);

save('/Users/michelleshu/Documents/Mitchell/CRNN-MEG/data/D/D_crnn_avrg.mat', 'data', 'time', 'words');
end

function rnnData = forward(data, rnn, params)
data = permute(data, [2 3 1]);
numWords = size(data, 3);

rnnData = zeros(params.numRNN, params.numFilters, numWords);
for r = 1 : params.numRNN
    if mod(r, 8)==0
        disp(['RNN: ' num2str(r)]);
    end
    
    tree = data;  
    
    % Layer 1
    WTemp = squeeze(rnn.WTemp(r, :, :));
    WPar = squeeze(rnn.WPar(r, :, :));
    WFront = squeeze(rnn.WFront(r, :, :));
    WVis = squeeze(rnn.WVis(r, :, :));
    
    treeTemp = tanh(WTemp * reshape(tree(:, [1 2], :), [], numWords));
    treePar = tanh(WPar * reshape(tree(:, [3 4], :), [], numWords));
    treeFront = tanh(WFront * reshape(tree(:, [5 6 9], :), [], numWords));
    treeVis = tanh(WVis * reshape(tree(:, [7 8 10], :), [], numWords));
    
    % Layer 2 (top layer)
    WTop = squeeze(rnn.WTop(r, :, :));
    tree = cat(3, treeTemp, treePar, treeFront, treeVis);
    tree = permute(tree, [1 3 2]);
    tree = tanh(WTop * reshape(tree, [], numWords));

    rnnData(r, :, :) = tree;    
end

rnnData = permute(rnnData, [3 1 2]);
end

function rnn = initRandomRNNWeights(params)
% R/L temporal to temporal cortex
rnn.WTemp = zeros(params.numRNN, params.numFilters, params.numFilters * 2);
% R/L parietal to parietal cortex
rnn.WPar = zeros(params.numRNN, params.numFilters, params.numFilters * 2);
% R/L/M frontal to frontal cortex
rnn.WFront = zeros(params.numRNN, params.numFilters, params.numFilters * 3);
% R/L/M visual to visual cortex
rnn.WVis = zeros(params.numRNN, params.numFilters, params.numFilters * 3);
% Top level: temporal, parietal, frontal, visual to whole brain
rnn.WTop = zeros(params.numRNN, params.numFilters, params.numFilters *4); 

for i = 1 : params.numRNN
    rnn.WTemp(i, :, :) = -0.11 + 0.22 * rand(params.numFilters, ...
                                             params.numFilters * 2); 
    rnn.WPar(i, :, :) = -0.11 + 0.22 * rand(params.numFilters, ...
                                            params.numFilters * 2); 
    rnn.WFront(i, :, :) = -0.11 + 0.22 * rand(params.numFilters, ...
                                              params.numFilters * 3);  
    rnn.WVis(i, :, :) = -0.11 + 0.22 * rand(params.numFilters, ...
                                            params.numFilters * 3); 
    rnn.WTop(i, :, :) = -0.11 + 0.22 * rand(params.numFilters, ...
                                            params.numFilters * 4);  
end
end