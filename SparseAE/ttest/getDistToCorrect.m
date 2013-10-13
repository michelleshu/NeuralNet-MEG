function [d11, d22] = getDistToCorrect(classifyFile)
%% Return 2 * 30 distances, between predictions and correct features

load(classifyFile);

num_folds=30;
dist_metric='cosine';

d11 = zeros(num_folds, 1);
d22 = zeros(num_folds, 1);

for i = 1:num_folds,    
    testWordNums = find(folds==i);
    cur_preds = ests(testWordNums,:);
    true_labs = targets(testWordNums,:); 
    
    d11(i) = pdist2(cur_preds(1,1:218~=13),true_labs(1,1:218~=13),dist_metric);
    d22(i) = pdist2(cur_preds(2,1:218~=13),true_labs(2,1:218~=13),dist_metric);   
end

end

