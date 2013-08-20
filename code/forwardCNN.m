function [ train, test ] = forwardCNN( filters, params )
% Generate the training and test sets of base CNN layer
% Adapted from Richard Socher's code

nf = params.numFilters;
fl = params.numRegions;    % final length of input feature representation
                           % equal to # of brain regions pooled into

numTest = 2;   % number of words to use for testing
labelNo = 76;   % index of semantic feature to be used as label


% train and test will contain nf x fl representations of words                                                 
train = struct('data', zeros(params.numWords - numTest, nf, fl), ...
        'labels', [], 'count', 0, 'words', []);
test = struct('data', zeros(numTest, nf, fl), 'labels', [], 'count', 0, ...
        'words', []);

word_features = extractFeaturesAllWords(filters, params);
word_ind = randperm(params.numWords);  % random ordering of word indices

% Copy out first two words to test set.
for i = 1 : numTest
    test.data(i, :, :) = word_features(word_ind(i), :, :);
    test.labels(i) = params.semMatrix(word_ind(i), labelNo);
    test.count = test.count + 1;
    test.words{i} = params.wordNames{word_ind(i)};
end

% Copy remaining words to training set.
for i = numTest + 1 : params.numWords
    train.data(i - numTest, :, :) = word_features(word_ind(i), :, :);
    train.labels(i - numTest) = params.semMatrix(word_ind(i), labelNo);
    train.count = train.count + 1;
    train.words{i - numTest} = params.wordNames{word_ind(i)};
end

