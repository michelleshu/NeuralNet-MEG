% predict the text vectors
% Set fCrossValidate to true to test different regularization params
%
% This function uses kernel ridge regression with a linear kernel. This allows us to use the full
% brain as input features because we avoid the large inversion of the voxels/voxels matrix

function [weightMatrix, r] = learn_text_from_fmri_kernel_sep_lambda_no_bias(trainingData, trainingTargets, fCrossValidate)
% append a vector of ones so we can learn weights and biases together
b0 = mean(trainingTargets);
% trainingTargets = trainingTargets - repmat(b0,size(trainingTargets,1),1);
numTargetDimensions = size(trainingTargets,2);
numExamples = size(trainingData,1);

params = [.0000001 .000001 .00001 .0001 .001 .01 .1 .5 1 5 10 50 100 500 1000 10000 20000 50000 ...
    100000 500000 1000000 5000000 10000000];
%params = [.00001 .0001 .001 .01 .1  1 10 100 1000 10000];
n_params = length(params);
n_targs = size(trainingTargets,2);

CVerr = zeros(n_params, n_targs);

if (fCrossValidate==1)
    
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
        weightMatrix = KP * trainingTargets;
        
        % Original code for none kernel version
        %%%%%%%%%%%%%%%%%%%%%%%%%%%
        %weightMatrix = (trainingData'*trainingData + regularizationParam*eye(numDimensions)) \
        %trainingData'* trainingTargets;
        % compute the cross validation error using Hastie CV trick
        %S = trainingData*inv(trainingData'*trainingData + regularizationParam*eye(numDimensions))* trainingData';
        %%%%%%%%%%%%%%%%%%%%%%%%%%%
        Snorm = repmat(1 - diag(S), 1, numTargetDimensions);
        YdiffMat = (trainingTargets - (trainingData*weightMatrix));
        YdiffMat = YdiffMat ./ Snorm;
        CVerr(i,:) = (1/numExamples).*sum(YdiffMat .* YdiffMat);
        %CVerrs = CVerrs ./ new_counts(1:numTargetDimensions); % targets are ordered by freq..normalize by word freq
        %[sortedErrs, sortedErrInd] = sort(CVerrs,2,'ascend');
        %targetIndices(i,:) = sortedErrInd(1:4500);
        %pause
        %         CVavgerr(i) = mean(CVerrs);
        %         CVmaxerr(i) = max(CVerrs);
        %         CVminerr(i) = min(CVerrs);
        %         %fprintf('lambda: %f CVAvgErr: %f, CVMinErr*1e8: %f, CVMaxErr: %f\n', regularizationParam, CVavgerr(i), CVminerr(i)*1e8, CVmaxerr(i));
        %         regParams(i) = regularizationParam;
    end
    
    % try using min of avg err
    [minerr, minerrIndex] = min(CVerr);
    r=zeros(1,n_targs);
    for cur_targ = 1:n_targs,
        regularizationParam = params(minerrIndex(cur_targ));
        r(cur_targ) = regularizationParam;
        %roundIndices = targetIndices(minerrIndex,:);
        %trainingTargets = trainingTargets(:,roundIndices); % select only the good targets
        %weightMatrix = (trainingData'*trainingData + regularizationParam*eye(numDimensions)) \
        %trainingData'* trainingTargets;
        
        % got good param, now obtain weights
        dlambda = D + regularizationParam*speye(size(D));
        dlambdaInv = diag(1 ./ diag(dlambda));
        KlambdaInv = V * dlambdaInv * U';
        
        % Solve for weight matrix so we can compute residual
        weightMatrix(:,cur_targ) = trainingData' * KlambdaInv * trainingTargets(:,cur_targ);
    end
    
else
    % solve a simultaneous least squares solution (i.e. multiple outputs)
    %fprintf('Input dimensions: %d, target dimensions %d\n', numDimensions, numTargetDimensions);
    %fprintf('Converting targets and solving L2 regularized least squares solution\n');
    regularizationParam = fCrossValidate;
    %  fprintf(1,'RegularizationParam = %f\n', regularizationParam);
    %weightMatrix = (trainingData'*trainingData + regularizationParam*eye(numDimensions)) \
    %trainingData'* trainingTargets;
    
    weightMatrix = kernel_ridge_regression(trainingData, trainingTargets, 'linear', regularizationParam);
    
    %   S = trainingData*inv(trainingData'*trainingData + regularizationParam*eye(numDimensions))* trainingData';
    %   Snorm = repmat(1 - diag(S), 1, numTargetDimensions);
    %   YdiffMat = (trainingTargets - trainingData*weightMatrix);
    %   YdiffMat = YdiffMat ./ Snorm;
    %   CVerrs = (1/numExamples).*sum(YdiffMat .* YdiffMat);
    %   CVerrs = CVerrs ./ new_counts(1:5000); % normalize errs by freq
end

weightMatrix(end+1,:) = b0;
return;

