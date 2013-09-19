% predict the text vectors

THIS CODE IS WRONG - handling the bias term wrong

function [weightMatrix, minerr] = learn_text_from_fmri_kernel_sep_lambda_no_bias_permute(trainingData, trainingTargets_in, n_permutes)

b0 = mean(trainingTargets_in);
trainingTargets = trainingTargets_in-b0;
numTargetDimensions = size(trainingTargets,2);
numExamples = size(trainingData,1);
numFeats = size(trainingData,2);

params = [.0000001 .000001 .00001 .0001 .001 .01 .1 .5 1 5 10 50 100 500 1000 10000 20000 50000 ...
    100000 500000 1000000 5000000 10000000];

n_params = length(params);
n_targs = size(trainingTargets,2);

CVerr_true = zeros(n_params, n_targs);
CVerr_perm = zeros(n_permutes, n_params, n_targs);

permute_labs = zeros([ n_permutes, size(trainingTargets)]);
for i = 1:n_permutes,
    permute_labs(i,:,:) = trainingTargets(randperm(numExamples),:);
end



% If we do an eigendecomp first we can quickly compute the inverse for many different values
% of lambda. SVD uses X = UDV' form.
% First compute K0 = (XX' + lambda*I) where lambda = 0.
K0 = trainingData*trainingData';
[U,D,V] = svd(K0);

for i = 1:length(params)
    regularizationParam = params(i);
    %    fprintf('CVLoop: Testing regularation param: %f, ', regularizationParam);
    
    
    % Now we can obtain Kinv for any lambda doing Kinv = V * (D + lambda*I)^-1 U'
    dlambda = D + regularizationParam*speye(size(D));
    dlambdaInv = diag(1 ./ diag(dlambda));
    KlambdaInv = V * dlambdaInv * U';
    
    % Compute pseudoinverse of linear kernel.
    KP = trainingData' * KlambdaInv;
    
    % Compute S matrix of Hastie Trick X*KP
    S = trainingData * KP;
    
    % Solve for weight matrix so we can compute residual
    weightMatrix = zeros(numFeats+1,numTargetDimensions);
    weightMatrix(1:end-1,:) = KP * trainingTargets;
    weightMatrix(end,:) = b0;
    Snorm = repmat(1 - diag(S), 1, numTargetDimensions);
    YdiffMat = (trainingTargets_in - (trainingData*weightMatrix));
    YdiffMat = YdiffMat ./ Snorm;
    CVerr_true(i,:) = (1/numExamples).*sum(YdiffMat .* YdiffMat);
    
    for j = 1:n_permutes,
        
        weightMatrix(1:end-1,:)= KP * permute_labs(j,:,:);
        weightMatrix(end,:) = b0;
        Snorm = repmat(1 - diag(S), 1, numTargetDimensions);
        YdiffMat = (trainingTargets - (trainingData*weightMatrix));
        YdiffMat = YdiffMat ./ Snorm;
        CVerr_perm(j,i,:) = (1/numExamples).*sum(YdiffMat .* YdiffMat);
    end
end

% try using min of avg err
[minerr, minerrIndex] = min(CVerr_true);

for cur_targ = 1:n_targs,
    regularizationParam = params(minerrIndex(cur_targ));
    
    % got good param, now obtain weights
    dlambda = D + regularizationParam*speye(size(D));
    dlambdaInv = diag(1 ./ diag(dlambda));
    KlambdaInv = V * dlambdaInv * U';
    
    % Solve for weight matrix so we can compute residual
    weightMatrix(1:end-1,cur_targ) = trainingData' * KlambdaInv * trainingTargets(:,cur_targ);
end

weightMatrix(end,:) = b0;
return;

