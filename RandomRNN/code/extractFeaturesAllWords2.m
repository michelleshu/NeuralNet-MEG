function features = extractFeaturesAllWords(filters, params)
% Given data matrix of all words (words x sensors x timepoints), transform 
% all into feature matrices. Return (words x filters x pooledtimes).

features = zeros(params.numWords, params.numFilters, ...
            params.numSensors * params.numTimeSections);

for i = 1 : params.numWords
    word = squeeze(params.data(i, :, :));
    features(i, :, :) = extractFeatures2(word, filters, params);
end 
end