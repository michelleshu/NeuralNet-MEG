function [experiment] = MultiGNBTest(experiment)
numExamples = size(experiment.testExamples,1);
numClasses = length(experiment.classNames);

logLikelihood = zeros(numClasses,numExamples);
for i = 1:numClasses,
    meanmat = repmat(experiment.trainingParams.mean(i,:), numExamples, 1);
    sigmamat = repmat(experiment.trainingParams.std(i,:), numExamples, 1);
    logLikelihood(i,:) = sum(computeLogLikelihood(experiment.testExamples, ...
        meanmat, sigmamat), 2);
    if ~isfield(experiment,'dropPrior') || experiment.dropPrior == 0,
        logLikelihood(i,:)  =  logLikelihood(i,:)  + log(experiment.trainingParams.p(1));
    end
end

labels = zeros(1,numExamples);
for i = 1:numExamples,
     %m=max(logLikelihood(:,i))
     %s=size(logLikelihood(:,i))
    %sum(logLikelihood(:,i) == max(logLikelihood(:,i)))
    labels(i) = experiment.classNames(logLikelihood(:,i) == max(logLikelihood(:,i)));
end



% sigma0 = sqrt(experiment.trainingParams.variance0);
% sigma1 = sqrt(experiment.trainingParams.variance1);
%
% %compute all likelihoods at once
% mean0mat = repmat(experiment.trainingParams.mean0, numExamples, 1);
% mean1mat = repmat(experiment.trainingParams.mean1, numExamples, 1);
% sigma0mat = repmat(sigma0, numExamples, 1);
% sigma1mat = repmat(sigma1, numExamples, 1);
%
% %matrix (or vector if 1 example) of individual likelihoods across features
% %likelihood0 = normpdf(experiment.testExamples, mean0mat, sigma0mat);
% %likelihood1 = normpdf(experiment.testExamples, mean1mat, sigma1mat);
%
% % REMOVE ME - stores scores
% %logLikelihood0 = computeLogLikelihood(experiment.testExamples, mean0mat, sigma0mat);
% %logLikelihood1 = computeLogLikelihood(experiment.testExamples, mean1mat, sigma1mat);
% %labelMatrix = logLikelihood1 > logLikelihood0;
% %trueLabelMatrix = repmat(experiment.testLabels, 1, numFeatures);
% %size(trueLabelMatrix)
% %featureErrs = ((labelMatrix - trueLabelMatrix).^2);
% %experiment.allTestCorrect(experiment.validation.round,:) = ~featureErrs;
% %experiment.allTestErrs(experiment.validation.round,:) = featureErrs;
% %size(experiment.allTestCorrect(experiment.validation.round,:))
% %pause
% %%%%
%
% logLikelihood0 = sum(computeLogLikelihood(experiment.testExamples, mean0mat, sigma0mat), 2);
% logLikelihood1 = sum(computeLogLikelihood(experiment.testExamples, mean1mat, sigma1mat), 2);

%we should work in log space. Sum across rows. This will leave us with a
%column vector
%ll0 = log(likelihood0);
%ll1 = log(likelihood1);
%ll0(find(ll0 == -Inf)) = 0; %fix the -Inf. We won't consider these in our score
%ll1(find(ll1 == -Inf)) = 0;
%logLikelihood0 = sum(ll0,2);
%logLikelihood1 = sum(ll1,2);
% scoreClass0 = logLikelihood0 + log(experiment.trainingParams.p0); %vector of scores for each example
% scoreClass1 = logLikelihood1 + log(experiment.trainingParams.p1);

%we can compute the actual probability
%even though we are in log space using the log-sum trick
%maxScoreClass = max(scoreClass0,scoreClass1);
%logTrickScoreClass0 = scoreClass0 - maxScoreClass;
%logTrickScoreClass1 = scoreClass1 - maxScoreClass;
%probClass0 = exp(logTrickScoreClass0) / ( exp(logTrickScoreClass0) + exp(logTrickScoreClass1) );
%probClass1 = exp(logTrickScoreClass1) / ( exp(logTrickScoreClass0) + exp(logTrickScoreClass1) );

%we can label all our test examples at once
% labels = scoreClass1 > scoreClass0;

%record result
testresult.numTests = numExamples;
testresult.trueLabels = experiment.testLabels;
testresult.classifierLabels = labels;
testresult.numCorrect = sum((testresult.trueLabels - testresult.classifierLabels) == 0);
experiment.numCorrect = testresult.numCorrect;
experiment.testresult = testresult;
%experiment.results(experiment.producer.currentStudy, experiment.producer.currentSubject, experiment.producer.currentROI, experiment.validation.round) = testresult;


