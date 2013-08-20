


subj_ids = {'A','B','C','D','E','F','G','I','J'};
load ~/sem_matrix.mat
sem_matrix = zscore(sem_matrix);
sem_length = size(sem_matrix,2);



res_dir = '~/research/fmri/results/feature_proj/l2/noz_data/by_phase/';
%res_dir = '/usr1/afyshe/fmri/results/l2/zscore_sem/by_phase';



feat_types  = {'cwave_haar'};%'fft_power','fft_phase'};

numwords=60;
num_folds =30;
num_trials = 5;
time_start = -0.1;
time_end=0.75;
window_length=0.05;
window_inds = window_length/0.005;


frq_cwt = scal2frq(1:64,'haar',0.005);
cwt_inds = {};
cwt_inds{1}= find(frq_cwt<4);
cwt_inds{2}= find(frq_cwt>=4 & frq_cwt< 8);
cwt_inds{3}= find(frq_cwt>=8 & frq_cwt< 13);
cwt_inds{4}= find(frq_cwt>=13 & frq_cwt< 30);
cwt_inds{5}= find(frq_cwt>=30 & frq_cwt< 50);

t_vec = time_start:window_length:time_end;

all_pove = zeros(length(feat_types),length(subj_ids),...
    length(t_vec),length(cwt_inds),...
    num_trials,218);

rank_all = zeros(length(feat_types),length(subj_ids),...
    length(t_vec),length(cwt_inds),...
    num_trials,60);
rank_1000_all = zeros(length(feat_types),length(subj_ids),...
    length(t_vec),length(cwt_inds),...
    num_trials,60);
rank_1000_dist = zeros(length(feat_types),length(subj_ids),...
    length(t_vec),length(cwt_inds),...
    num_trials,60);
results_2v2 = zeros(length(feat_types),length(subj_ids),...
    length(t_vec),length(cwt_inds),...
    num_trials,num_folds);

for s = 1:length(subj_ids),
    subj = subj_ids{s};
    fprintf('Cur subj %s %s\n',subj, datestr(now));
    res_sub_dir = sprintf('%s/%s',res_dir,subj);
    if ~exist(res_sub_dir,'dir'),
        mkdir(res_sub_dir);
    end
    
    %files have data structures with the following fields
    %'data': the output of the feature transformation
    %'time': the time points for the beginning of each feature window.  this vector will have the same length as the last dimension of 'data'.
    %'window_width': the width of each feature window (i.e. the window for which the given feature was created)
    %'window_step': the step size between adjacent windows.  if window_step = window_width then the windows do not overlap at all.
    %'words': the order of the words as they appear in 'data'.
    for f_type = 1:length(feat_types),
        
        for t_ind = 1:length(t_vec),
            t=t_vec(t_ind);
            fprintf('t %.3f... ',t);
            for f = 1:length(cwt_inds),
                cur_fs = cwt_inds{f};
                
                
                for tr = 1:num_trials,
                    
                    load(sprintf('%s/%s_%s_t%.3f-%.3f_f%.2f-%.2f_%i.mat',...
                        res_sub_dir,subj,feat_types{f_type},t,t+window_length,frq_cwt(cur_fs(1)),frq_cwt(cur_fs(end)),tr));
                    
                    all_pove(f_type,s,t_ind,f,tr,:) =  (1-mean((ests - sem_matrix).^2)./mean((sem_matrix-repmat(mean(sem_matrix),[60,1])).^2));
                    
                    for i = 1:num_folds,
                        
                        testWordNums = find(folds==i);
                        
                        if length(testWordNums) ~= 2,
                            die ;
                        end
                        
                        cur_preds = ests(testWordNums,:);
                        true_labs = sem_matrix(testWordNums,:);
                        
                        for cur_w=1:2,
                            
                            d = pdist2(cur_preds(cur_w,1:218~=13),sem_matrix(:,1:218~=13),'cosine');
                            [y,inds]=sort(d(1:60));
                            rank_all(f_type,s,t_ind,f,tr,testWordNums(cur_w)) = find(inds==testWordNums(cur_w));
                            
                            sem_1000cat = cat(1,sem_matrix(testWordNums(cur_w),1:218~=13), sem_mat1000(:,1:218~=13));
                            
                            
                            d=pdist2(cur_preds(cur_w,1:218~=13),sem_1000cat,'cosine');
                            [y,inds]=sort(d);
                            rank_1000_all(f_type,s,t_ind,f,tr,testWordNums(cur_w)) = find(inds==1);
                            rank_1000_dist(f_type,s,t_ind,f,tr,testWordNums(cur_w)) = d(1);
                        end
                        
                        d = pdist2(cur_preds,true_labs,'cosine');
                        if (d(1,1)+d(2,2)) < (d(1,2)+d(2,1)),
                            results_2v2( f_type,s,t_ind,f,tr,i) = 1;
                        end
                        
                    end
                    
                    
                end
            end
            fprintf('\n');
        end
        
    end
end



%%


load Intel218Questions


load ~/sem_matrix.mat
[sem_matrix,mu,sigma] = zscore(sem_matrix);
i1000 = load('~/bagOfFeatures.mat');

sem_mat1000 = (i1000.features(:,1:218)-3)/2;
sem_mat1000 = (sem_mat1000 - repmat(mu,[size(sem_mat1000,1),1]))./repmat(sigma,[size(sem_mat1000,1),1]);
sem_mat1000(1:60,:) = [];

m = 1-squeeze(mean(mean(mean(rank_1000_all,6),5),2))/941;

%imagesc(m')
im2 = imresize(m, 8, 'nearest');
%  pcolor(im2' )
%  shading flat;
imagesc(im2')
axis tight
title(sprintf('Percentage Rank Accuracy (Higher is better)'),'fontsize',14);
set(gca,'xTick',[1:length(t_vec)]*8-3)
set(gca,'xticklabel',t_vec,'fontsize',14);
set(gca,'yTick',([1:5])*8-3)
set(gca,'yticklabel',{'Delta','Theta','Alpha','Beta','Gamma'},'fontsize',14);
ylabel('Frequency band','fontsize',14)
xlabel('Time','fontsize',14)
colorbar('fontsize',14);
save2pdf(sprintf('~/research/fmri/graphs/feat_proj/l2_zscore_by_phase_time_rank.pdf'));



%%



m = squeeze(mean(mean(mean(results_2v2,6),5),2));

%imagesc(m')
im2 = imresize(m, 8, 'nearest');
%  pcolor(im2' )
%  shading flat;
imagesc(im2')
axis tight
title(sprintf('2 vs 2 Accuracy'),'fontsize',16);
set(gca,'xTick',[1:length(t_vec)]*8-3)
set(gca,'xticklabel',t_vec,'fontsize',16);
set(gca,'yTick',([1:5])*8-3)
set(gca,'yticklabel',{['Delta' char(10) '(0-4Hz)'],'Theta (4-8Hz)','Alpha (8-13Hz)','Beta (13-30Hz)','Gamma (>30Hz)'},'fontsize',16);
%my_xticklabels(gca,'yticklabel',{['Delta' char(10) '(0-4Hz)'],'Theta (4-8Hz)','Alpha (8-13Hz)','Beta (13-30Hz)','Gamma (>30Hz)'},'fontsize',16);

ylabel('Frequency band','fontsize',16)
xlabel('Time (0 = stimulus onset)','fontsize',16)
cbar_axes =colorbar('fontsize',16);
c=get(cbar_axes);
set(get(cbar_axes,'ylabel'),'String', '2 vs 2 Accuracy','fontsize',16);
save2pdf(sprintf('~/research/fmri/graphs/feat_proj/l2_zscore_by_phase_time_2vs2.pdf'));

%%

band_names = {'Delta','Theta','Alpha','Beta','Gamma'};
% two way anova
r = squeeze(rank_1000_dist(1,:,:,:,:));

r = permute(r,[2 3 1 4]);
% indexes are now time, freq band, subject, distances
s = size(r);
%[p,table,stats] = anovan(r(:,:)',prod(s(3:4)),'model','interaction');

ind = 1;
gs = cell(numel(r));
gs2 = cell(numel(r));
for i = 1:size(r,1),
    for j = 1:size(r,2),
        for k = 1:size(r(:,:,:),3),
            gs{ind} = band_names(j);
            gs2{ind} = num2str(t_vec(i));
            ind = ind+1;
        end
    end
end


[p,table,stats] = anovan(r(:)',{gs gs2},'model','interaction');
[c,m] = multcompare(stats);

len = size(r,1);

b = zeros(len,len);
for i = 1:size(c,1),
    b(c(i,1),c(i,2)) = sign(c(i,3))== sign(c(i,5));
    b(c(i,2),c(i,1)) = sign(c(i,3))== sign(c(i,5));
end

%%

m = squeeze(mean(mean(all_pove,5),2));

maxm = 0;
for i = 1:218%[1 9 6],%,
    cur_m = m(:,:,i)';
    if max(cur_m(:)) > maxm,
        maxm=max(cur_m(:));
    end
end

all_meta = [];
for fi = 1:length(fld),
    all_meta  = [all_meta  meta.(fld{fi})];
    
end

for i = all_meta%[1 9 6],%,
    cur_m = m(:,:,i)';
    if (sum(cur_m(:)>0.1) <1)
        continue;
    end
    %cur_m(cur_m<0)=0;
    %min(cur_m(:))
    cur_min = min(cur_m(:));
    %     surf(t_vec, 1:5, cur_m,'EdgeColor','none');
    % view(0,90);
    % set(gcf,'Renderer','Zbuffer')
    % axis tight
    
    imagesc(cur_m,[cur_min,maxm])
    
    axis tight
    title(sprintf('Frequency Time POVE, Haar Wavelet, Q:%s',Intel218Questions{i}));
    set(gca,'xTick',[1:length(t_vec)])
    set(gca,'xticklabel',t_vec,'fontsize',11);
    set(gca,'yTick',[1:5])
    set(gca,'yticklabel',{'Delta','Theta','Alpha','Beta','Gamma'},'fontsize',11);
    ylabel('Frequency band')
    xlabel('Time')
    colorbar;
    save2pdf(sprintf('~/research/fmri/graphs/feat_proj/l2_zscore_by_phase_time_%i.pdf',i));
    %waitforbuttonpress
end
