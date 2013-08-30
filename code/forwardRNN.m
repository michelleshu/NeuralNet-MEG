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

save(params.out_file, 'data', 'time', 'words');
end

function rnnData = forward(data, rnn, params)
data = permute(data, [2 3 1]);

rnnData = zeros(params.numRNN, params.numFilters, params.numWords);
for r = 1 : params.numRNN
    tree = data;  
    
    % Layer 3 (time sections combine to full timeseries in each sub-region)
    if (params.numTimeSections > 1)
        WTimeSections = squeeze(rnn.WTimeSections(r, :, :));

        treeBRTimes = zeros(params.numRegions, params.numFilters, params.numWords);
        for br = 1 : params.numRegions
            br_indices = (1 : params.numRegions : (params.numRegions * ... 
                params.numTimeSections));
            treeBRTime = tanh(WTimeSections * ...
                reshape(tree(:, br_indices, :), [], params.numWords));
            treeBRTimes(br, :, :) = treeBRTime;
        end
        tree = permute(treeBRTimes, [2 1 3]);
    end
    
    % Layer 2 (sub-regions of brain to major regions)
    WTemp = squeeze(rnn.WTemp(r, :, :));
    WPar = squeeze(rnn.WPar(r, :, :));
    WFront = squeeze(rnn.WFront(r, :, :));
    WVis = squeeze(rnn.WVis(r, :, :));
    
    treeTemp = tanh(WTemp * reshape(tree(:, [1 2], :), [], params.numWords));
    treePar = tanh(WPar * reshape(tree(:, [3 4], :), [], params.numWords));
    treeFront = tanh(WFront * reshape(tree(:, [5 6 9], :), [], params.numWords));
    treeVis = tanh(WVis * reshape(tree(:, [7 8 10], :), [], params.numWords));
    
    % Layer 1 (top layer)
    WTop = squeeze(rnn.WTop(r, :, :));
    tree = cat(3, treeTemp, treePar, treeFront, treeVis);
    
    tree = permute(tree, [1 3 2]);
    tree = tanh(WTop * reshape(tree, [], params.numWords));

    rnnData(r, :, :) = tree;    
end

disp([num2str(params.numRNN) ' trained']);

rnnData = permute(rnnData, [3 1 2]);
end

function rnn = initRandomRNNWeights(params)
% Sections of time for each sub-region combine
rnn.WTimeSections = zeros(params.numRNN, params.numFilters, ...
    params.numFilters * params.numTimeSections);

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
    rnn.WTimeSections(i, :, :) = -0.11 + 0.22 * rand(params.numFilters, ...
                               params.numFilters * params.numTimeSections);
    
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