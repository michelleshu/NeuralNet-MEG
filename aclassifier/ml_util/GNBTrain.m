
function [experiment] = GNBTrain(experiment)
numLabels = length(experiment.trainLabels);
numFeatures = size(experiment.trainExamples,2);

class0Ind = find(experiment.trainLabels == 0);
class1Ind = find(experiment.trainLabels == 1);

numClass0 = length(class0Ind);
numClass1 = length(class1Ind);
Class0 = experiment.trainExamples(class0Ind,:);
Class1 = experiment.trainExamples(class1Ind,:);

%calc probability of each class
experiment.trainingParams.p0 = numClass0/numLabels;
experiment.trainingParams.p1 = numClass1/numLabels;

%calc means 
experiment.trainingParams.mean0 = mean(Class0,1);
experiment.trainingParams.mean1 = mean(Class1,1);

%calc variances
if (numClass0 > 1)
  experiment.trainingParams.variance0 = var(Class0);
else
  experiment.trainingParams.variance0 = ones(1,numFeatures); %if one example set to standard normal variance
end

if (numClass1 > 1)
  experiment.trainingParams.variance1 = var(Class1);
else
  experiment.trainingParams.variance1 = ones(1,numFeatures);
end
  

