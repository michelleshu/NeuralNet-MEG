% Get features with mean value close to 0.5
count = 0;
eligibleIndices = zeros(218, 1);
for i = 1 : 218
    if (means(i) > 0.3 && means(i) < 0.7)
        eligibleIndices(count + 1) = i;
        count = count + 1;
    end
end

eligibleValues = zeros(count, 1);
for i = 1 : count
    eligibleValues(i) = percentCorrectSAE(eligibleIndices(i));
end

[sortedVals, sortedInds] = sort(eligibleValues);

% Change indices back to the original semantic feature indices
for i = 1 : count
    sortedInds(i) = eligibleIndices(sortedInds(i));
end
