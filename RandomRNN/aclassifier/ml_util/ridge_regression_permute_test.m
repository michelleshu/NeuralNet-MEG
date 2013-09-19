% Solve for weightMatrix using a ridge regression penalty (L2).
% Simultaneously perform numPermutes permutation tests.
% Also uses LOOCV to tune regularization parameter
%
% Input:
% trainingData: independent variables (X) (matrix N*P)
% trainingTargets: dependent variables (Y) (vector or matrix N*C)
% numPermutes: number of permutation tests to perform
%
% Output:
% weightMatrix: weight matrix as
%
%
%weightMatrix, minerr, minerr_perm
function [pove, pove_perm, loocv_preds,loocv_preds_perm] = ridge_regression_permute_test(trainingData, trainingTargets, numPermutes)
% append a vector of ones so we can learn weights and biases together
trainingData(:,end+1) = 1;  % we shouldn't use this bias with L2 reg.
numDimensions = size(trainingData,2);
numExamples = size(trainingData,1);

params = [.0000001 .000001 .00001 .0001 .001 .01 .1 .5 1 5 10 50 100 500 1000 10000 20000 50000 ...
    100000 500000 1000000 5000000 10000000];
numParams = length(params);
numTargs = size(trainingTargets,2);

permuteLabs = zeros([numExamples, numTargs, numPermutes]);
for i = 1:numPermutes,
    permuteLabs(:,:,i) = trainingTargets(randperm(numExamples),:);
end

loocv_preds = zeros(numExamples,numTargs);
loocv_preds_perm = zeros(numExamples,numTargs,numPermutes);

for i = 1:numExamples,
    if rem(i,2)==1,
        fprintf('Fold %i (%s)...',i,datestr(now));
    end
    folds = 1:numExamples ~= i;
    [pred, pred_perm] = tune_regularizer(trainingData(folds,:), ...
        trainingTargets(folds,:), permuteLabs(folds,:,:),trainingData(~folds,:), params);
    loocv_preds(i,:) = pred;
    loocv_preds_perm(i,:,:) = pred_perm;
end
fprintf('\n');

pove = 1 - mean((loocv_preds - trainingTargets).^2)./var(trainingTargets,1);
pove_perm = zeros(numTargs,numPermutes);
for i = 1:numPermutes,
    pove_perm(:,i) = 1 - mean((squeeze(loocv_preds_perm(:,:,i)) - trainingTargets).^2)./var(trainingTargets,1);
end

return;
end

function [pred, pred_perm] = tune_regularizer(trainingData, trainingTargs, permuteTrainTargs, testData, params)

numPermutes = size(permuteTrainTargs,3);
numDimensions = size(trainingData,2);
numExamples = size(trainingData,1);
numParams = length(params);
numTargs = size(trainingTargs,2);


CVerr_true = zeros(numParams, numTargs);
CVerr_perm = zeros(numPermutes,numParams,numTargs);

pove_all_true = zeros(numParams, numTargs);
pove_all_perm = zeros(numPermutes,numParams,numTargs);

targ_var = var(trainingTargs,1);
K0 = trainingData*trainingData';
[U,D,V] = svd(K0);

for i = 1:length(params)
    regularizationParam = params(i);
    
    % Now we can obtain Kinv for any lambda doing Kinv = V * (D + lambda*I)^-1 U'
    dlambda = D + regularizationParam*eye(size(D));
    dlambdaInv = diag(1 ./ diag(dlambda));
    KlambdaInv = V * dlambdaInv * U';
    
    % Compute pseudoinverse of linear kernel.
    KP = trainingData' * KlambdaInv;
    
    % Compute S matrix of Hastie Trick X*KP
    S = trainingData * KP;
    
    % Solve for weight matrix so we can compute residual
    weightMatrix = KP * trainingTargs;
    Snorm = repmat(1 - diag(S), 1, numTargs);
    preds = trainingData*weightMatrix;
    YdiffMat = (trainingTargs - preds);
    YdiffMat = YdiffMat ./ Snorm;
    CVerr_true(i,:) = (1/numExamples).*sum(YdiffMat .* YdiffMat);
    pove_all_true(i,:) = 1- CVerr_true(i,:)./var(trainingTargs,1);
    %Now do it for permuted labels too.
    %     weightMatrix = multiprod(KP, permuteTrainTargs);
    %     perm_preds = multiprod(trainingData,weightMatrix);
    perm_preds = zeros(numExamples, numTargs, numPermutes);
    for j = 1:numPermutes,
        weightMatrix = KP * permuteTrainTargs(:,:,j);
        perm_preds(:,:,j) = trainingData*weightMatrix;
        %         YdiffMat = (permuteTrainTargs(:,:,j) - perm_preds(:,:,j));
        YdiffMat = (permuteTrainTargs(:,:,j) - perm_preds(:,:,j));
        YdiffMat = YdiffMat ./ Snorm;
        cur_cverr = (1/numExamples).*sum(YdiffMat .* YdiffMat);
        CVerr_perm(j,i,:) = cur_cverr;
        pove_all_perm(j,i,:) = 1- cur_cverr./targ_var;
    end
end



[~, minerrIndex] = min(CVerr_true);
pove=zeros(1,numTargs);
weightMatrix = zeros(numDimensions,numTargs);

for cur_targ = 1:numTargs,
    pove(cur_targ) =pove_all_true(minerrIndex(cur_targ),cur_targ);
    regularizationParam = params(minerrIndex(cur_targ));
    
    % got good param, now obtain weights
    dlambda = D + regularizationParam*eye(size(D));
    dlambdaInv = diag(1 ./ diag(dlambda));
    KlambdaInv = V * dlambdaInv * U';
    
    KP = trainingData' * KlambdaInv;
    
    % Solve for weight matrix so we can compute residual
    weightMatrix(:,cur_targ) = KP * trainingTargs(:,cur_targ);
end
pred = testData * weightMatrix;

pred_perm = zeros(numTargs,numPermutes);
%weightPerm = zeros(numDimensions,numTargs,numPermutes );
for j = 1:numPermutes,
    [~, minerrIndex_perm] =min(squeeze(CVerr_perm(j,:,:)));
    for cur_targ = 1:numTargs,
        regularizationParam = params(minerrIndex_perm(cur_targ));
        
        % got good param, now obtain weights
        dlambda = D + regularizationParam*eye(size(D));
        dlambdaInv = diag(1 ./ diag(dlambda));
        KlambdaInv = V * dlambdaInv * U';
        
        KP = trainingData' * KlambdaInv;
        
        % Solve for weight matrix so we can compute residual
        weightMatrix(:,cur_targ) = KP * permuteTrainTargs(:,cur_targ,j);
        
    end
    pred_perm(:,j) = testData * weightMatrix;
end


return;
end


