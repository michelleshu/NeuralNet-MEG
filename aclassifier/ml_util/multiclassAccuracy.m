function [ accuracy ] = multiclassAccuracy( all_data, labels )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
%fprintf('mca %s\n',datestr(now))
numFolds = size(all_data,1);
folds = 1:numFolds;
total_correct = zeros(1,numFolds);
for i = 1:numFolds,
    experiment.trainLabels = labels(folds~= i);
    
    experiment.trainExamples = all_data(folds ~= i, :);%(second dim is number of features)
    
    experiment.testLabels = labels(folds == i);
    experiment.testExamples = all_data(folds == i, :);
    experiment.dropPrior=0;
    experiment = MultiGNBTrain(experiment);
    experiment = MultiGNBTest(experiment);
    total_correct( i) = experiment.numCorrect/sum(folds == i);
    
end
accuracy = mean(total_correct);
end

