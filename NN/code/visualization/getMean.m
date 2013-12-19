hiddenIndices = [1, 12, 13];
averageInput = zeros(306, 30);
valsToAverage = origInputs(hiddenIndices, :, :);

for i = 1 : 306
    for j = 1 : 30
        averageInput(i, j) = mean(squeeze(valsToAverage(:, i, j)));
    end
end