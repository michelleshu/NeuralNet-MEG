function features = extractFeaturesAllWords(filters, params)
% Given data matrix of all words (words x sensors x timepoints), transform 
% all into feature matrices. Return (words x filters x pooledtimes).

features = zeros(size(params.data, 1), params.numFilters, ...
            numel(fields(params.brainRegions)));

for i = 1 : size(params.data, 1)
    word = squeeze(params.data(i, :, :));
    features(i, :, :) = extractFeatures(word, filters, params);
end 
end