

load Intel218Questions


load ~/sem_matrix.mat
[sem_matrix,mu,sigma] = zscore(sem_matrix);
i1000 = load('~/bagOfFeatures.mat');

sem_mat1000 = (i1000.features(:,1:218)-3)/2;
sem_mat1000 = (sem_mat1000 - repmat(mu,[size(sem_mat1000,1),1]))./repmat(sigma,[size(sem_mat1000,1),1]);
sem_mat1000(1:60,:) = [];


%%  to load results: run rank_pove first

i1000 = load('~/bagOfFeatures.mat');


load ~/sem_matrix.mat
sem_matrix = zscore(sem_matrix);

% sem_mat1000 = (i1000.features(:,1:218)-3)/2;
% sem_mat1000(1:60,:) = sem_matrix;
% sem_mat1000 =zscore(sem_mat1000);
% sem_matrix = sem_mat1000(1:60,:);

do_2v2=1;

num_words=60;
num_trials = 5;
num_folds=30;
all_preds = all_ests;
avrg_res = zeros(2,length(feat_types));

all_rank = zeros(length(feat_types),length(subj_ids));
all_twos = zeros(length(feat_types),length(subj_ids));
all_wons = zeros(length(feat_types),length(subj_ids));

rank_all = zeros(length(feat_types),length(subj_ids),num_trials,60);
rank_1000_all = zeros(length(feat_types),length(subj_ids),num_trials,60);
rank_1000_all_top10 = zeros(length(feat_types),length(subj_ids),num_trials,60,10);
rank_1000_dist = zeros(length(feat_types),length(subj_ids),num_trials,60);
rank_1000_all_naive = zeros(length(feat_types),length(subj_ids),num_trials,60);

for f = 1:length(feat_types),
    
    for s = 1:length(subj_ids),
        rank = zeros(num_trials,60);
        rank_1000 = zeros(num_trials,60);
        twos = zeros(num_trials,30);
        wons = zeros(num_trials,2,30);
        for t = 1:num_trials,
            dists = zeros(2,2,30);
            folds = all_folds(f,s,t,:);
            
            for i = 1:num_folds,
                
                testWordNums = find(folds==i);
                
                %                 if length(testWordNums) ~= 2,
                %                     die ;
                %                 end
                
                cur_preds = squeeze(all_preds(f,s,t,testWordNums,:));
                true_labs = sem_matrix(testWordNums,:);
                
                if do_2v2==1,
                    d11 = pdist2(cur_preds(1,1:218~=13),true_labs(1,1:218~=13),'cosine');
                    d22 = pdist2(cur_preds(2,1:218~=13),true_labs(2,1:218~=13),'cosine');
                    
                    d12 = pdist2(cur_preds(1,1:218~=13),true_labs(2,1:218~=13),'cosine');
                    d21 = pdist2(cur_preds(2,1:218~=13),true_labs(1,1:218~=13),'cosine');
                    
                    
                    dists(1,1,i)=d11;
                    dists(1,2,i)=d12;
                    dists(2,1,i)=d21;
                    dists(2,2,i)=d22;
                    
                    if d11 + d22 <= d12 + d21,
                        twos(t,i) = 1;
                    end
                    if d11 <= d12,
                        wons(t,1,i) = 1;
                    end
                    
                    if d22 <= d21,
                        wons(t,2,i) = 1;
                    end
                end
                naive_pred = mean(sem_matrix(folds~=i,1:218~=13));
                for cur_w=1:length(testWordNums),
                    %                     d = pdist(cat(1,cur_preds(cur_w,1:218~=13), sem_matrix(:,1:218~=13)));
                    %                     [y,inds]=sort(d(1:60));
                    %                     rank(t,testWordNums(cur_w)) = find(inds==testWordNums(cur_w));
                    %
                    %                     d = sqrt(sum((repmat(cur_preds(cur_w,1:218~=13),[940,1]) - sem_mat1000(:,1:218~=13)).^2,2));
                    %
                    %                     [y,inds]=sort(d);
                    %                     rank_1000(t,testWordNums(cur_w)) = find(inds==testWordNums(cur_w));
                    
                    
                    
                    %d = pdist(cat(1,cur_preds(cur_w,1:218~=13), sem_matrix(:,1:218~=13)),'cosine');
                    if do_2v2==1,
                        d = pdist2(cur_preds(cur_w,1:218~=13),sem_matrix(:,1:218~=13),'cosine');
                        [y,inds]=sort(d(1:60));
                        rank(t,testWordNums(cur_w)) = find(inds==testWordNums(cur_w));
                        rank_all(f,s,t,testWordNums(cur_w)) = find(inds==testWordNums(cur_w));
                    end
                    sem_1000cat = cat(1,sem_matrix(testWordNums(cur_w),1:218~=13), sem_mat1000(:,1:218~=13));
                    
                    %d = (sum(repmat(cur_preds(cur_w,1:218~=13),[941,1]) .* sem_1000cat,2))./...
                    %    sqrt(sum((repmat(cur_preds(cur_w,1:218~=13),[941,1])).^2,2)).* sqrt(sum((sem_1000cat).^2,2));
                    if do_2v2==1,
                        d=pdist2(cur_preds(cur_w,1:218~=13),sem_1000cat,'cosine');
                    else 
                        d=pdist2(cur_preds(1:218~=13),sem_1000cat,'cosine');
                    end
                    [y,inds]=sort(d(1:941));
                    rank_1000(t,testWordNums(cur_w)) = find(inds==1);
                    rank_1000_all(f,s,t,testWordNums(cur_w)) = find(inds==1);
                    rank_1000_dist(f,s,t,testWordNums(cur_w)) = d(1);
                    rank_1000_all_top10(f,s,t,testWordNums(cur_w),:)=inds(1:10);
                    
                    d=pdist2(naive_pred,sem_1000cat,'cosine');
                    [y,inds]=sort(d(1:941));
                    rank_1000_all_naive(f,s,t,testWordNums(cur_w)) = find(inds==1);
                end
                
            end
            %             fprintf('\n********\nFeat %s\nSubj %s\n2x2 %.4f\n1v2 %.4f\nMean rank:%.2f\nMedian rank:%.2f\n********\n',...
            %                 feat_types{f}, subj,mean(twos(:)),mean(wons(:)), mean(rank), median(rank) );
            %             avrg_res(1,f) = avrg_res(1,f) + mean(twos(:));
            %             avrg_res(2,f) = avrg_res(2,f) + mean(wons(:));
            
        end
        fprintf('\n********\nFeat %s\nSubj %s\n2x2 %.4f\n1v2 %.4f\nMean rank:%.2f\nMedian rank:%.2f\nMean rank 1000:%.2f\nMedian rank 1000:%.2f\n********\n',...
            feat_types{f}, subj_ids{s} ,mean(twos(:)),mean(wons(:)),...
            mean(rank(:)), median(rank(:)),mean(rank_1000(:)), median(rank_1000(:)) );
        all_rank(f,s) = mean(rank(:));
        all_wons(f,s) = mean(wons(:));
        all_twos(f,s) = mean(twos(:));
    end
    fprintf('\n********\nFeat %s\n2x2 %.4f\n1v2 %.4f\nMean rank:%.2f\n********\n',...
        feat_types{f}, mean(all_twos(f,:)),mean(all_wons(f,:)),...
        mean(all_rank(f,:)));
end

for f = 1:length(feat_types),
    fprintf('\n********\nFeat %s\n2x2 %.4f\n1v2 %.4f\nMean rank:%.2f\n********\n',...
        feat_types{f}, mean(all_twos(f,:)),mean(all_wons(f,:)),...
        mean(all_rank(f,:)));
end

%%
for i=1:length(feat_types),
    fprintf('%s & ',feat_types_printable_short{i});
    for j = 1:size(all_twos,2),
        fprintf(' %.1f & ', all_twos(i,j)*100);
    end
    fprintf('%.1f \\\\ \n',mean(all_twos(i,:))*100);
end

%%

for i=1:length(feat_types),
    fprintf('%s & ',feat_types_printable_short{i});
    for j = 1:size(all_twos,2),
        fprintf(' %.1f & ', all_rank(i,j));
    end
    fprintf('%.1f  \\\\ \n',mean(all_rank(i,:)));
end

%%
for k =4,
    fprintf('\n\n\n\nRound %i\n',k)
feat_types_printable_short={'Magnitude','Windowed Mean','Phase','Power','Wavelets'};
for i=1:length(feat_types),
    fprintf('%s & ',feat_types_printable_short{i});
    for j = 1:length(subj_ids),
        fprintf(' %.1f & ', (1-median(rank_1000_all(i,j,k,:))/941)*100);
    end
    fprintf('%.1f  \\\\ \n',mean(median((1-rank_1000_all(i,:,k,:)/941)*100,4)));
end
end

%%

feat_types_printable_short={'Magnitude','Windowed Mean','Phase','Power','Wavelets'};

feat_types_printable_short=feat_types;
for i=1:length(feat_types),
    fprintf('%s & ',feat_types_printable_short{i});
    for j = 1:length(subj_ids),
        fprintf(' %.1f & ', (1-median(rank_1000_all(i,j,:))/941)*100);
    end
    fprintf('%.1f  \\\\ \n',mean(median((1-rank_1000_all(i,:,:)/941)*100,3)));
end
end

%%
for i=1:length(feat_types),
    fprintf('%s & ',feat_types_printable_short{i});
    for j = 1:length(subj_ids),
        fprintf(' %.1f & ', (1-median(rank_1000_all_naive(i,j,:))/941)*100);
    end
    fprintf('%.1f  \\\\ \n',median((1-rank_1000_all_naive(i,:)/941)*100));
end

%%