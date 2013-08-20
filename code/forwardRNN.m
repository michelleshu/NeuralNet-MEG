function [train test] = forwardRNN(train, test, params)
% Take final output from CNN and stack an RNN on top.
% At each level, children will be pooled according to progressively more 
% general brain regions.

rnn = initRandomRNNWeights(params);

% Forward prop training data
disp('Forward Prop Train...');
% Output: numTrain x numRNN x numHid
train.data = forward(train.data, rnn, params);

end

function rnnData = forward(data, rnn, params)
data = permute(data, [2 3 1]);
numWords = numel(data, 3);

rnnData = zeros(params.numRNN, params.numFilters, numWords);
for r = 1 : numRNN
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
    
    % Layer 2
    WTop = squeeze(rnn.WTop(r, :, :));
    rnnData(r, :, :) = squeeze(tree);
    
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