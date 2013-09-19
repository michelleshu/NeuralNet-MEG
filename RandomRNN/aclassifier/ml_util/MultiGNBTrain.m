
function [experiment] = MultiGNBTrain(experiment)
numLabels = length(experiment.trainLabels);
numFeatures = size(experiment.trainExamples,2);
classNames = unique(experiment.trainLabels);
numClasses = length(classNames);

classInds = false(numClasses,numLabels);
classSize = zeros(1,numClasses);
experiment.trainingParams.p = zeros(1,numClasses);
experiment.trainingParams.mean = zeros(numClasses,numFeatures);
experiment.trainingParams.std = zeros(numClasses,numFeatures);

for i = 1:numClasses,
    cur_class = classNames(i);
    classInds(i,:) = logical(experiment.trainLabels == cur_class);
    classSize(i) = sum(classInds(i,:));
    experiment.trainingParams.p(i) = classSize(i)/numLabels;
    experiment.trainingParams.mean(i,:);
    mean(experiment.trainExamples(classInds(i,:),:));
    experiment.trainingParams.mean(i,:) = ...
        mean(experiment.trainExamples(classInds(i,:),:));
    if (classSize(i) > 1)
        experiment.trainingParams.std(i,:) = ...
            std(experiment.trainExamples(classInds(i,:),:));
    else
        experiment.trainingParams.std(i,:) = 1;
    end
    experiment.trainingParams.std(i,experiment.trainingParams.std(i,:)==0) = 10^-10;
end
experiment.classNames = classNames;

end


