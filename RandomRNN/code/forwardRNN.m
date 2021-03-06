function forwardRNN(filters, params)
% Take final output from CNN and stack an RNN on top.
% At each level, children will be pooled according to progressively more 
% general brain regions.

% Extract features from raw data
data = extractFeaturesAllWords(filters, params);
time = params.time;
words = params.wordNames;

% Forward prop training data
disp('Forward Propagating through RNNs...');
% Output: numTrain x numRNN x numHid
data = forward(data, params);

save(params.out_file, 'data', 'time', 'words');
end

function rnnData = forward(data, params)
data = permute(data, [2 3 1]);

rnnData = zeros(params.numRNN, params.numFilters * params.numTimeSections, params.numWords);
for r = 1 : params.numRNN
    % Reserve space for all time sections of top level vector
    top_vector = zeros(params.numFilters * params.numTimeSections, params.numWords);
    for timeSection = 1 : params.numTimeSections
        rnn = initRandomRNNWeights(params);
        
        tree = data(:, (timeSection - 1) * params.numRegions + 1 : ...
            timeSection * params.numRegions, :);
    
        % Layer 2 (sub-regions of brain to major regions)
        treeTemp = tanh(rnn.WTemp * reshape(tree(:, [1 2], :), [], params.numWords));
        treePar = tanh(rnn.WPar * reshape(tree(:, [3 4], :), [], params.numWords));
        treeFront = tanh(rnn.WFront * reshape(tree(:, [5 6 9], :), [], params.numWords));
        treeVis = tanh(rnn.WVis * reshape(tree(:, [7 8 10], :), [], params.numWords));
        
        % Layer 1 (top layer)
        tree = cat(3, treeTemp, treePar, treeFront, treeVis);

        tree = permute(tree, [1 3 2]);
        tree = tanh(rnn.WTop * reshape(tree, [], params.numWords));
        
        top_vector((timeSection - 1) * params.numFilters + 1 : ...
            timeSection * params.numFilters, :) = tree;
    end
    rnnData(r, :, :) = top_vector;
end

disp([num2str(params.numRNN) ' trained']);

rnnData = permute(rnnData, [3 1 2]);
end

function rnn = initRandomRNNWeights(params)
    % R/L temporal to temporal cortex
    rnn.WTemp = zeros(params.numFilters, params.numFilters * 2);
    % R/L parietal to parietal cortex
    rnn.WPar = zeros(params.numFilters, params.numFilters * 2);
    % R/L/M frontal to frontal cortex
    rnn.WFront = zeros(params.numFilters, params.numFilters * 3);
    % R/L/M visual to visual cortex
    rnn.WVis = zeros(params.numFilters, params.numFilters * 3);
    % Top level: temporal, parietal, frontal, visual to whole brain
    rnn.WTop = zeros(params.numFilters, params.numFilters *4); 

    rnn.WTemp(:, :) = -0.11 + 0.22 * rand(params.numFilters, ...
                                                 params.numFilters * 2); 
    rnn.WPar(:, :) = -0.11 + 0.22 * rand(params.numFilters, ...
                                                params.numFilters * 2); 
    rnn.WFront(:, :) = -0.11 + 0.22 * rand(params.numFilters, ...
                                                  params.numFilters * 3);  
    rnn.WVis(:, :) = -0.11 + 0.22 * rand(params.numFilters, ...
                                                params.numFilters * 3);
    rnn.WTop(:, :) = -0.11 + 0.22 * rand(params.numFilters, ...
                                                params.numFilters * 4);  
end