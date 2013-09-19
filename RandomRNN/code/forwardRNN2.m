function forwardRNN(filters, params)
% Take final output from CNN and stack an RNN on top.
% At each level, children will be pooled according to progressively more 
% general brain regions.

% Extract features from raw data
data = extractFeaturesAllWords2(filters, params);
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
        
        
        
        tree = data(:, (timeSection - 1) * params.numSensors + 1 : ...
            timeSection * params.numSensors, :);
        
        % Layer 3 (individual sensors to sub-regions of brain)
        treeLTemp = tanh(rnn.WLTemp * reshape(tree(:, ...
            params.brainRegions.LeftTemporal, :), [], params.numWords));
        treeRTemp = tanh(rnn.WRTemp * reshape(tree(:, ...
            params.brainRegions.RightTemporal, :), [], params.numWords));
        treeLPar = tanh(rnn.WLPar * reshape(tree(:, ...
            params.brainRegions.LeftParietal, :), [], params.numWords));
        treeRPar = tanh(rnn.WRPar * reshape(tree(:, ...
            params.brainRegions.RightParietal, :), [], params.numWords));
        treeLFront = tanh(rnn.WLFront * reshape(tree(:, ...
            params.brainRegions.LeftFrontal, :), [], params.numWords));
        treeRFront = tanh(rnn.WRFront * reshape(tree(:, ...
            params.brainRegions.RightFrontal, :), [], params.numWords));
        treeLVis = tanh(rnn.WLVis * reshape(tree(:, ...
            params.brainRegions.LeftVisual, :), [], params.numWords));
        treeRVis = tanh(rnn.WRVis * reshape(tree(:, ...
            params.brainRegions.RightVisual, :), [], params.numWords));
        treeMFront = tanh(rnn.WMFront * reshape(tree(:, ...
            params.brainRegions.MidlineFrontal, :), [], params.numWords));
        treeMVis = tanh(rnn.WMVis * reshape(tree(:, ...
            params.brainRegions.MidlineVisual, :), [], params.numWords));
    
        % Layer 2 (sub-regions of brain to major regions)
        treeTemp = permute(cat(3, treeLTemp, treeRTemp), [1 3 2]);
        treeTemp = tanh(rnn.WTemp * reshape(treeTemp, [], params.numWords));
        
        treePar = permute(cat(3, treeLPar, treeRPar), [1 3 2]);
        treePar = tanh(rnn.WPar * reshape(treePar, [], params.numWords));
        
        treeFront = permute(cat(3, treeLFront, treeRFront, treeMFront), [1 3 2]);
        treeFront = tanh(rnn.WFront * reshape(treeFront, [], params.numWords));
        
        treeVis = permute(cat(3, treeLVis, treeRVis, treeMVis), [1 3 2]);
        treeVis = tanh(rnn.WVis * reshape(treeVis, [], params.numWords));
        
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
    % Specific brain regions
    rnn.WLTemp = zeros(params.numFilters, params.numFilters * 36);
    rnn.WRTemp = zeros(params.numFilters, params.numFilters * 36);
    rnn.WLPar = zeros(params.numFilters, params.numFilters * 39);
    rnn.WRPar = zeros(params.numFilters, params.numFilters * 39);
    rnn.WLFront = zeros(params.numFilters, params.numFilters * 33);
    rnn.WRFront = zeros(params.numFilters, params.numFilters * 33);
    rnn.WLVis = zeros(params.numFilters, params.numFilters * 36);
    rnn.WRVis = zeros(params.numFilters, params.numFilters * 36);
    rnn.WMFront = zeros(params.numFilters, params.numFilters * 12);
    rnn.WMVis = zeros(params.numFilters, params.numFilters * 6);
    
    rnn.WLTemp(:, :) = -0.11 + 0.22 * rand(params.numFilters, ...
                                                 params.numFilters * 36); 
    rnn.WRTemp(:, :) = -0.11 + 0.22 * rand(params.numFilters, ...
                                                 params.numFilters * 36); 
    rnn.WLPar(:, :) = -0.11 + 0.22 * rand(params.numFilters, ...
                                                 params.numFilters * 39);                                          
    rnn.WRPar(:, :) = -0.11 + 0.22 * rand(params.numFilters, ...
                                                 params.numFilters * 39); 
    rnn.WLFront(:, :) = -0.11 + 0.22 * rand(params.numFilters, ...
                                                 params.numFilters * 33); 
    rnn.WRFront(:, :) = -0.11 + 0.22 * rand(params.numFilters, ...
                                                 params.numFilters * 33); 
    rnn.WLVis(:, :) = -0.11 + 0.22 * rand(params.numFilters, ...
                                                 params.numFilters * 36);                                          
    rnn.WRVis(:, :) = -0.11 + 0.22 * rand(params.numFilters, ...
                                                 params.numFilters * 36);    
    rnn.WMFront(:, :) = -0.11 + 0.22 * rand(params.numFilters, ...
                                                 params.numFilters * 12); 
    rnn.WMVis(:, :) = -0.11 + 0.22 * rand(params.numFilters, ...
                                                 params.numFilters * 6);                                                                             
                                             
    % R/L temporal to temporal cortex
    rnn.WTemp = zeros(params.numFilters, params.numFilters * 2);
    % R/L parietal to parietal cortex
    rnn.WPar = zeros(params.numFilters, params.numFilters * 2);
    % R/L/M frontal to frontal cortex
    rnn.WFront = zeros(params.numFilters, params.numFilters * 3);
    % R/L/M visual to visual cortex
    rnn.WVis = zeros(params.numFilters, params.numFilters * 3);
    % Top level: temporal, parietal, frontal, visual to whole brain
    rnn.WTop = zeros(params.numFilters, params.numFilters * 4); 

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