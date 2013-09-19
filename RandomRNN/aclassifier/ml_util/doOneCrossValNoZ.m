function [ests, ranks, preds, pove] = doOneCrossValNoZ(cur_data, sem_matrix, folds, numFolds, numWords, save_file,zscore_folds,dist_metric)
%Given the data, semantic matrix, cross validation folds, number of
%folds, number of words and the file into which the results should be
%saved, perform cross validation.  Save results to file.

num_sem = size(sem_matrix,2);
preds = zeros(numWords);
targets = zeros(size(sem_matrix));
ranks = zeros(1,numWords);
ests = zeros(numWords,num_sem);
for i = 1:numFolds,
    %fprintf('Fold %i... ',i);
    %     cur_sem_test = sem_matrix(folds==i,:);
    %     cur_sem_train = sem_matrix(folds~=i,:);
    %     if zscore_folds ==1,
    %         [cur_sem_train,mu,sigma] = zscore(cur_sem_train);
    %         cur_sem_test = (sem_matrix(folds==i,:)-repmat(mu,sum(folds==i),1))./repmat(sigma,sum(folds==i),1);
    %     end
    %
    %     cur_sem_all = zeros(size(sem_matrix));
    %     cur_sem_all(folds~=i,:) = cur_sem_train;
    %     cur_sem_all(folds==i,:) = cur_sem_test;
    % targets(folds==i,:) = cur_sem_test;
    %    [weightMatrix,r]=learn_text_from_fmri_kernel_sep_lambda(cur_data(folds~=i,:),...
    %         cur_sem_train,1);
    
    cur_sem_all=sem_matrix;
    targets(folds==i,:) =sem_matrix(folds==i,:);
%     [weightMatrix,r]=learn_text_from_fmri_kernel_sep_lambda(cur_data(folds~=i,:),...
%         sem_matrix(folds~=i,:),1);
%     
     [weightMatrix,r]=learn_text_from_fmri_kernel_sep_lambda_no_bias(cur_data(folds~=i,:),...
        sem_matrix(folds~=i,:),1);
%    
% 
%     
%      [weightMatrix]=learn_text_from_fmri_kernel_sep_lambda_no_bias(cur_data(folds~=i,:),...
%         sem_matrix(folds~=i,:),0);
% r=0;
%     
%     [weightMatrix,r]=learn_text_from_fmri_kernel_sep_lambda_no_bias_centered(cur_data(folds~=i,:),...
%         sem_matrix(folds~=i,:),1);
%     
    

   % fprintf('r:%f\n',median(r));
    
    testBrainExample=cur_data(folds==i,:);
    testWordNums = find(folds==i);
    % append a vector of ones for the biases
    testBrainExample(:,end+1) = 1;
    
    for cur_word = 1:size(testBrainExample,1),
        
        targetEstimate = testBrainExample(cur_word,:)*weightMatrix;
        ests(testWordNums(cur_word),:) = targetEstimate;
        d = pdist([targetEstimate; cur_sem_all], dist_metric);
        [y,ind] = sort(d(1:numWords));
        ranks(testWordNums(cur_word)) = find(testWordNums(cur_word)==ind);
        preds(testWordNums(cur_word),:) = ind;
    end
        
end
pove = (1-mean((ests - targets).^2)./mean((targets-repmat(mean(targets),[numWords,1])).^2));
save(save_file,'ests','ranks','preds','folds','pove','targets');
return;
end