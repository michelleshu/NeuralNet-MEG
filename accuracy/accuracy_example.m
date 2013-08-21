
% load('/Users/michelleshu/Documents/Mitchell/CRNN-MEG/results/D/D_crnn_5.mat');
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

fprintf('1 vs 2 accuracy %.2f\n',100*mean(wons(:)));
fprintf('2 vs 2 accuracy %.2f\n',100*mean(twos));
