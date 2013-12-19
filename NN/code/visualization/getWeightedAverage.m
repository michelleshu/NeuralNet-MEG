% Get weighted average ideal input that results in activation of output
% node representing a particular feature
function averageInput = getWeightedAverage(W2, origInputs, featureIndex)

weights = squeeze(W2(featureIndex, :));

averageInput = zeros(306, 30);

for i = 1 : 306
    for j = 1 : 30
        for h = 1 : numel(weights)
            averageInput(i, j) = averageInput(i, j) + ...
                origInputs(h, i, j) * weights(h);
        end
    end
end

averageInput = averageInput ./ numel(weights);