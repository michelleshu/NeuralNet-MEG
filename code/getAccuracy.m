function [acc_ones, acc_twos] = getAccuracy(filename)

load(filename);
% will load
% folds: [60x1 double]
% ests: [60x218 double]
% ranks: [1x60 double]
% preds: [60x60 double]
% pove: [1x218 double]
% targets: [60x218 double]


num_folds=30;
dist_metric='cosine';

dists = zeros(2,2,30);
twos = zeros(1,30);
wons = zeros(2,30);

for i = 1:num_folds,
    
    testWordNums = find(folds==i);
    
    %                 if length(testWordNums) ~= 2,
    %                     die ;
    %                 end
    
    cur_preds = ests(testWordNums,:);
    true_labs = targets(testWordNums,:);
    
    
    
    d11 = pdist2(cur_preds(1,1:218~=13),true_labs(1,1:218~=13),dist_metric);
    d22 = pdist2(cur_preds(2,1:218~=13),true_labs(2,1:218~=13),dist_metric);
    
    d12 = pdist2(cur_preds(1,1:218~=13),true_labs(2,1:218~=13),dist_metric);
    d21 = pdist2(cur_preds(2,1:218~=13),true_labs(1,1:218~=13),dist_metric);
    
    
    dists(1,1,i)=d11;
    dists(1,2,i)=d12;
    dists(2,1,i)=d21;
    dists(2,2,i)=d22;
    
    if d11 + d22 <= d12 + d21,
        twos(i) = 1;
    end
    if d11 <= d12,
        wons(1,i) = 1;
    end
    
    if d22 <= d21,
        wons(2,i) = 1;
    end
    
end

acc_ones = 100*mean(wons(:));
acc_twos = 100*mean(twos);
