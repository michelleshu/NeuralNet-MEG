

load Intel218Questions

load ~/sem_matrix.mat
[sem_matrix,mu,sigma] = zscore(sem_matrix);
i1000 = load('~/bagOfFeatures.mat');

sem_mat1000 = (i1000.features(:,1:218)-3)/2;
sem_mat1000 = (sem_mat1000 - repmat(mu,[size(sem_mat1000,1),1]))./repmat(sigma,[size(sem_mat1000,1),1]);
sem_mat1000(1:60,:) = [];

%%
feat_set = 'all';
%feat_types  = {'raw','gradioms','magnetoms','gp','wmean','wslope','fft_power','fft_phase','cwave_haar'};%,'corr'};%,'corr','cwave_haar'};
feat_types  = {'raw','wmean','fft_power','fft_phase','cwave_haar'};%,'corr'};%,'corr','cwave_haar'};
subj_ids = {'A','B','C','D','E','F','G','I','J'};%,'B','C','F','G','I'};

% feat_set = 'raw_phase_combo';
% feat_types  = {'raw','fft_phase','raw_fft_phase'};%,'corr'};%,'corr','cwave_haar'};
% subj_ids = {'A','B','C'};

num_words=60;
num_trials = 5;
num_folds=30;
nperm=12;

res_dir = '~/research/fmri/results/feature_proj/l2/';
res_dir = '~/research/fmri/results/feature_proj/l2/redo_perms_noz/';

lsdir = dir(sprintf('%sz*PERMUTE_*73*',res_dir));
lsdir = dir(sprintf('%s*PERMUTE_*',res_dir));
%lsdir = dir(sprintf('%sz*PERMUTE_*73493764422',res_dir));

%lsdir = dir(sprintf('%sn*PERMUTE_*',res_dir));

clear all_data
all_data(1:length(feat_types)) = struct('all_preds',zeros(2,num_trials,num_words,218),...
    'all_targets',zeros(2,num_trials,num_words,218),'all_pove',zeros(2,num_trials,218),...
    'folds',zeros(2,num_trials,60),'cur_ind',1);

for cdir = 1:length(lsdir),
    cur_dir = lsdir(cdir).name;
    fprintf('Now process dir %s\n',cur_dir);
    load(sprintf('%s%s/params.mat',res_dir,cur_dir));
    
    for s = 1:length(subj_ids),
        subj=subj_ids{s};
        if exist(sprintf('%s/%s/%s/',res_dir,cur_dir,subj),'dir'),
            
            for f = 1:length(feat_types),
                cur_feat = feat_types{f};
                for perm = 1:nperm,
                    % results without permute before average have this
                    % pattern
                    %cur_fdir = sprintf('%s/%s/%s/*%s*permute_%i_*.mat',res_dir,cur_dir,subj,feat_types{f},perm);
                    if strcmp(cur_feat,'cwave_haar')~=1
                        cur_fdir = sprintf('%s/%s/%s/*%s*_%i.mat',res_dir,cur_dir,subj,feat_types{f},perm);
                    else
                        % _1*-*.mat
                        cur_fdir = sprintf('%s/%s/%s/*%s_*_%i_1*-*.mat',res_dir,cur_dir,subj,feat_types{f},perm);
                    end
                    fdir = dir(cur_fdir);
                    %fprintf('len %i name %s\n',length(fdir),cur_fdir)
                    if (strcmp(cur_feat,'cwave_haar')~=1 && length(fdir) == num_trials) || ...
                            (strcmp(cur_feat,'cwave_haar')==1 && length(fdir) == 2*num_trials),
                        good_files = 0;
                        for i = 1:length(fdir),
                            fname = sprintf('%s%s/%s/%s',...
                                res_dir,cur_dir,subj_ids{s},fdir(i).name);
                            f_stats = dir(fname);
                            if f_stats.bytes > 0,
                                good_files = good_files+1;
                            end
                        end
                        if good_files == num_trials || good_files == 2*num_trials,
                            for i = 1:length(fdir),
                                fname = sprintf('%s%s/%s/%s',...
                                    res_dir,cur_dir,subj_ids{s},fdir(i).name);
                                if exist(fname,'file'),
                                    
                                    load(fname)
                                    inds = 1:218;
                                    if strcmp(cur_feat,'cwave_haar')==1,
                                        if ~isempty(strfind(fname,'end') ),
                                            inds = 101:218;
                                        else
                                            inds = 1:100;
                                        end
                                    end
                                    
                                    all_data(f).all_preds(all_data(f).cur_ind,i,:,inds) =  ests;
                                    all_data(f).all_targets(all_data(f).cur_ind,i,:,inds) =  targets;
                                    all_data(f).all_pove(all_data(f).cur_ind,i,inds) =...
                                        (1-mean((ests - targets).^2)./mean((targets-repmat(mean(targets),[60,1])).^2));
                                    all_data(f).all_folds(all_data(f).cur_ind,i,:) = folds;
                                    
                                else
                                    fprintf('Warning, cannot find file %s/%s/%s\n',cur_dir,subj_ids{s},fdir(i).name);
                                    die;
                                end
                            end
                            fprintf('Successfully loaded %s/%s/%s et al\n',cur_dir,subj_ids{s},fdir(1).name);
                            all_data(f).cur_ind = all_data(f).cur_ind+1;
                        else
                            if ~isempty(fdir),
                                fprintf('Incomplete trials %i %s\n',good_files,cur_fdir);
                            end
                            continue;
                        end
                    end
                    %     fprintf('subj %s %i\n',subj_ids{s}, sum(all_pove>0))
                    %     Intel218Questions(find(all_pove/5>0))
                end
            end
        end
    end
end

%%

avrg_res = zeros(2,length(feat_types));

%all_rank = zeros(length(feat_types),length(subj_ids));
all_twos_rand = zeros(length(feat_types),108);
all_wons_rand = zeros(length(feat_types),108);
all_rank_rand = zeros(length(feat_types),108,num_trials,60);
all_twos_rand(:) = NaN;
all_wons_rand(:) = NaN;
for f = 1:length(feat_types),
    fprintf('%s\n',feat_types{f});
    mrank = zeros(1,all_data(f).cur_ind-1);
    mrank_1000 = zeros(1,all_data(f).cur_ind-1);
    mwons = zeros(1,all_data(f).cur_ind-1);
    mtwos = zeros(1,all_data(f).cur_ind-1);
    rank_all = zeros(all_data(f).cur_ind-1,num_trials,60);
    rank_1000_all_rand = zeros(all_data(f).cur_ind-1,num_trials,60);
    for cind = 1:all_data(f).cur_ind-1,
        if rem(cind,10)==1,
            fprintf('%s %i...\n',datestr(now),cind);
        end
        
        rank = zeros(num_trials,60);
        rank_1000 = zeros(num_trials,60);
        twos = zeros(num_trials,30);
        wons = zeros(num_trials,2,30);
        for t = 1:num_trials,
            %             twos = zeros(1,30);
            %             wons = zeros(2,30);
            %             boths = zeros(1,30);
            dists = zeros(2,2,30);
            folds = squeeze(all_data(f).all_folds(cind,t,:));
            
            %             fname = sprintf('%s/%s/%s_%s_%i_permute_1_.mat',res_dir,subj_ids{s},subj_ids{s},feat_types{f},t);
            %             if exist(fname,'file'),
            %
            %                 load(fname)
            %                 all_preds(f,s,i,:,:) =  ests;
            %             else
            %                 fprintf('Warning, cannot find file %s\n',fname);
            %                 die;
            %             end
            
            for i = 1:num_folds,
                
                testWordNums = find(folds==i);
                
                if length(testWordNums) ~= 2,
                    die ;
                end
                
                cur_preds = squeeze(all_data(f).all_preds(cind,t,testWordNums,:));
                true_labs = squeeze(all_data(f).all_targets(cind,t,testWordNums,:));
                
                d11 = sqrt(sum((cur_preds(1,:) - true_labs(1,:)) .^ 2));
                d22 = sqrt(sum((cur_preds(2,:) - true_labs(2,:)) .^ 2));
                
                d12 = sqrt(sum((cur_preds(1,:) - true_labs(2,:)) .^ 2));
                d21 = sqrt(sum((cur_preds(2,:) - true_labs(1,:)) .^ 2));
                
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
                
                for cur_w=1:2,
%                     d = pdist(cat(1,cur_preds(cur_w,:), squeeze(all_data(f).all_targets(cind,t,:,:))));
%                     [y,inds]=sort(d(1:60));
%                     rank(t,testWordNums(cur_w)) = find(inds==testWordNums(cur_w));
%                     
%                     d = sqrt(sum((repmat(cur_preds(cur_w,1:218~=13),[940,1]) - sem_mat1000(:,1:218~=13)).^2,2));
%                     
%                     [y,inds]=sort(d);
%                     rank_1000(t,testWordNums(cur_w)) = find(inds==testWordNums(cur_w));
                    
                    %d = pdist(cat(1,cur_preds(cur_w,1:218~=13), sem_matrix(:,1:218~=13)),'cosine');
                    d = pdist2(cur_preds(cur_w,1:218~=13),sem_matrix(:,1:218~=13),'cosine');
                    [y,inds]=sort(d(1:60));
                    rank(t,testWordNums(cur_w)) = find(inds==testWordNums(cur_w));
                    rank_all(cind,t,testWordNums(cur_w)) = find(inds==testWordNums(cur_w));
                    
                    sem_1000cat = cat(1,sem_matrix(testWordNums(cur_w),1:218~=13), sem_mat1000(:,1:218~=13));
                    
                    %d = (sum(repmat(cur_preds(cur_w,1:218~=13),[941,1]) .* sem_1000cat,2))./...
                    %    sqrt(sum((repmat(cur_preds(cur_w,1:218~=13),[941,1])).^2,2)).* sqrt(sum((sem_1000cat).^2,2));
                    d=pdist2(cur_preds(cur_w,1:218~=13),sem_1000cat,'cosine');
                    [y,inds]=sort(d);
                    rank_1000(t,testWordNums(cur_w)) = find(inds==1);
                    rank_1000_all_rand(cind,t,testWordNums(cur_w)) = find(inds==1);
                    
                end
                
            end
            %             fprintf('\n********\nFeat %s\nSubj %s\n2x2 %.4f\n1v2 %.4f\nMean rank:%.2f\nMedian rank:%.2f\n********\n',...
            %                 feat_types{f}, subj,mean(twos(:)),mean(wons(:)), mean(rank), median(rank) );
            %             avrg_res(1,f) = avrg_res(1,f) + mean(twos(:));
            %             avrg_res(2,f) = avrg_res(2,f) + mean(wons(:));
            
        end
        %         fprintf('\n********\nFeat %s\nPermute %i\n2x2 %.4f\n1v2 %.4f\nMean rank:%.2f\nMedian rank:%.2f\nMean rank 1000:%.2f\nMedian rank 1000:%.2f\n********\n',...
        %             feat_types{f}, cind ,mean(twos(:)),mean(wons(:)),...
        %             mean(rank(:)), median(rank(:)),mean(rank_1000(:)), median(rank_1000(:)) );
        %         all_rank(f,s) = mean(rank(:));
        all_wons_rand(f,cind) = mean(wons(:));
        all_twos_rand(f,cind) = mean(twos(:));
        all_rank_rand(f,cind,:,:) = rank_1000;
        mrank_1000(cind) = mean(rank_1000(:));
        mrank(cind) = mean(rank(:));
        mwons(cind) = mean(wons(:));
        mtwos(cind) = mean(twos(:));
    end
    fprintf('\n********\nFeat %s\n2x2 %.4f (std %.4f max %.4f)\n1v2 %.4f (std %.4f max %.4f)\nMean rank:%.2f (std %.4f min %.4f)\nMedian rank:%.2f\nMean rank 1000:%.2f (std %.4f min %.4f)\nMedian rank 1000:%.2f\n********\n',...
        feat_types{f}, mean(mtwos),std(mtwos),max(mtwos),...
        mean(mwons),std(mwons),max(mwons),...
        mean(mrank),std(mrank),min(mrank),median(rank_all(:)),...
        mean(mrank_1000),std(mrank_1000),min(mrank_1000),median(rank_1000_all_rand(:)));
    subplot(1,2,1)
    hist(mtwos,20);
    subplot(1,2,2)
    hist(mwons,20);
    drawnow;
end





%%