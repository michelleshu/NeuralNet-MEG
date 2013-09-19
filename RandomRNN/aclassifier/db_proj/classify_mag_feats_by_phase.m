%
%
% % to create a classifiers based on features
%
% base_dir = '/usr1/afyshe/fmri/dbproj/';
% subj_ids = {'a','b','c','d','e','f','g','i','j'};
% feat_types  = {'raw','gp','wmean','wslope','fft_power','fft_phase'};
% for j = 1:length(subj_ids),
%     subj = subj_ids{j};
%
%     fprintf('subj %s\n',subj);
%     load ~/sem_matrix.mat
%
%
%     numwords=60;
%     numfolds =30;
%
%
%     folds = crossvalind('kfold',numwords,numfolds);
%     load(sprintf('%s%s_spec_feats.mat',base_dir,subj));
%
%     fprintf('spec ang %s\n', datestr(now));
%     cur_data = double(spec_ang);
%     doonecrossval(cur_data,sem_matrix, folds, numfolds, numwords, sprintf('%s/%s_spec_ang_class_results.mat',base_dir,subj));
%
%     fprintf('spec mag %s\n', datestr(now));
%     cur_data = double(spec_mag);
%     doonecrossval(cur_data,sem_matrix, folds, numfolds, numwords, sprintf('%s/%s_spec_mag_class_results.mat',base_dir,subj));
%
%
%
%     fprintf('spec both %s\n', datestr(now));
%     cur_data = double(cat(2,spec_ang,spec_mag));
%     doonecrossval(cur_data,sem_matrix, folds, numfolds, numwords, sprintf('%s/%s_spec_class_results.mat',base_dir,subj));
%
%     load(sprintf('%s%s_wave_feats.mat',base_dir,subj));
%
%     fprintf('wave data haar %s\n', datestr(now));
%     cur_data = double(cwave_haar);
%     doonecrossval(cur_data,sem_matrix, folds, numfolds, numwords, sprintf('%s/%s_harr_wave_class_results.mat',base_dir,subj));
%
%
%     fprintf('wave data morl %s\n', datestr(now));
%     cur_data = double(cwave_morl);
%     doonecrossval(cur_data,sem_matrix, folds, numfolds, numwords, sprintf('%s/%s_morl_wave_class_results.mat',base_dir,subj));
% end


subj_ids = {'D'};%,'G','I','J'};%'A','B','C','D','E','F','G','I','J'};
zscore_data = 0;


load ~/sem_matrix.mat
sem_matrix = zscore(sem_matrix);
sem_length = size(sem_matrix,2);

outfile_suffix = '_feat_corr.mat';


% res_dir = '~/research/fmri/results/l2/zscore_sem/';
% feats_dir = '~/research/fmri/data/20questions/features/';


% res_dir = '/usr1/afyshe/fmri/results/l2/noz_data/by_phase';
% feats_dir = '/usr1/afyshe/fmri/data/20questions/features/';

res_dir = '/other/bdstore01x/afyshe/fmri/results/l2/noz_data/by_phase/';
feats_dir = '~/fmri/data/20questions/features/';

% res_dir = '~/fmri/results/l2/zscore_sem/by_phase';
% feats_dir = '~/fmri/data/20questions/features/';

if ~exist(res_dir,'dir'),
    mkdir(res_dir);
end



feat_types  = {'cwave_haar'};%'fft_power','fft_phase'};

%{'corr'};%'raw','gp','wmean','wslope','fft_power','fft_phase'};%,'corr','cwave_haar'};

zscore_folds=0;
numwords=60;
numfolds =30;
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
cwt_inds{5}= find(frq_cwt>=30 & frq_cwt< 60);

save(sprintf('%s/params.mat',res_dir),'numwords','numfolds','num_trials',...
    'zscore_folds','time_start','time_end','window_length','window_inds','sem_matrix','frq_cwt','cwt_inds');


for s = 1:length(subj_ids),
    subj = subj_ids{s};
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
        cur_feat = feat_types{f_type};
        
        cur_fname = sprintf('%s/%s/%s_%s_avrg.mat',feats_dir,subj,subj,feat_types{f_type});
        if exist(cur_fname,'file') == 0,
            fprintf('warning - file does not exist: %s\n',cur_fname);
            continue;
        end
        
        fprintf('now load %s\n',cur_fname);
        struct = load(cur_fname);
        time = struct.time;
        struct.data = struct.data(:,:,:,time>=time_start & time <= time_end+window_length);
        if zscore_data ==1,
            struct.data = zscore(struct.data);
        end
        
        struct.time = time(time>=time_start & time <= time_end+window_length);
        time = struct.time;
        words = struct.words;
        
        for t = time_start:window_length:time_end,
            for f = 1:length(cwt_inds),
                cur_fs = cwt_inds{f};
                frq_start = frq_cwt(cur_fs(1));
                frq_end = frq_cwt(cur_fs(end));
                s_data = struct.data(:,:,cur_fs,:);
                s_size= size(s_data);
                num_sensors = prod(s_size(2:end-1));
                
                
                time = struct.time;
                t_ind = time >=t & time <t+window_length;
                time = struct.time(t_ind);
                time_length = length(time);
                
                cur_data = reshape(s_data(:,:,:,t_ind),[s_size(1),num_sensors,time_length]);
                
                %cur_data = double(cur_data);
                
                for tr = 1:num_trials,
                    fprintf('Subj %s time %.3f freq %i trial %i, %s\n',subj, t, f, tr,datestr(now));
                    rs = RandStream('mcg16807', 'Seed',tr);
                    RandStream.setDefaultStream(rs);
                    %rand('seed',tr);
                    folds = crossvalind('kfold',numwords,numfolds);
                    doOneCrossValNoZ(cur_data,sem_matrix, folds, numfolds, numwords, sprintf('%s/%s_%s_t%.3f-%.3f_f%.2f-%.2f_%i.mat',...
                        res_sub_dir,subj,cur_feat,t,t+window_length,frq_start,frq_end,tr),zscore_folds);
                    
                end
            end
        end
        
    end
end
