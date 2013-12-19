function maxAct = getMaxActivationSeries(featureIndex)

subjects = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'J'};
time = 0:25:725;
numComponents = 30;
timeSize = 30;

maxAct = zeros(length(subjects), timeSize);

for s = 1 : length(subjects)
    subject = subjects{s};

    % Load pre-computed network weights
    load(sprintf('./code/visualization/weights/%s_weights_z.mat', subject));

    optInputs = getOptHiddenNodeInputs(W1);
    origInputs = reversePCAInput(subject, optInputs, numComponents, timeSize);
    data = getWeightedAverage(W2, origInputs, featureIndex);
    
    maxAct(s, :) = max(data);
    
    % Scale each row independently
    maxAct(s, :) = maxAct(s, :) - min(maxAct(s, :));
    maxAct(s, :) = maxAct(s, :) ./ max(maxAct(s, :));
end



imagesc(time, [1:9], maxAct);