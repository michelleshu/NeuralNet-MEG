%
%
% % To create a classifiers based on features
%
% base_dir = '/usr1/afyshe/fmri/dbproj/';
% subj_ids = {'A','B','C','D','E','F','G','I','J'};
% feat_types  = {'raw','gp','wmean','wslope','fft_power','fft_phase'};
% for j = 1:length(subj_ids),
%     subj = subj_ids{j};
%
%     fprintf('Subj %s\n',subj);
%     load ~/sem_matrix.mat
%
%
%     numWords=60;
%     numFolds =30;
%
%
%     folds = crossvalind('Kfold',numWords,numFolds);
%     load(sprintf('%s%s_spec_feats.mat',base_dir,subj));
%
%     fprintf('Spec ang %s\n', datestr(now));
%     cur_data = double(spec_ang);
%     doOneCrossVal(cur_data,sem_matrix, folds, numFolds, numWords, sprintf('%s/%s_spec_ang_class_results.mat',base_dir,subj));
%
%     fprintf('Spec mag %s\n', datestr(now));
%     cur_data = double(spec_mag);
%     doOneCrossVal(cur_data,sem_matrix, folds, numFolds, numWords, sprintf('%s/%s_spec_mag_class_results.mat',base_dir,subj));
%
%
%
%     fprintf('Spec both %s\n', datestr(now));
%     cur_data = double(cat(2,spec_ang,spec_mag));
%     doOneCrossVal(cur_data,sem_matrix, folds, numFolds, numWords, sprintf('%s/%s_spec_class_results.mat',base_dir,subj));
%
%     load(sprintf('%s%s_wave_feats.mat',base_dir,subj));
%
%     fprintf('Wave data haar %s\n', datestr(now));
%     cur_data = double(cwave_haar);
%     doOneCrossVal(cur_data,sem_matrix, folds, numFolds, numWords, sprintf('%s/%s_harr_wave_class_results.mat',base_dir,subj));
%
%
%     fprintf('Wave data morl %s\n', datestr(now));
%     cur_data = double(cwave_morl);
%     doOneCrossVal(cur_data,sem_matrix, folds, numFolds, numWords, sprintf('%s/%s_morl_wave_class_results.mat',base_dir,subj));
% end

time_start=0;
time_end=0.75;

subj_ids = {'G','I','J'};%'A','B','C'};%,'D','E','F','G','I','J'};
load ~/sem_matrix.mat
sem_matrix = zscore(sem_matrix);
sem_length = size(sem_matrix,2);

% 
% res_dir = '~/research/fmri/results/l2/zscore_sem/';
% feats_dir = '~/research/fmri/data/20questions/features/';


res_dir = '/usr1/afyshe/fmri/results/l2/zscore_sem/redo_perms_yesbias/';%'/usr1/afyshe/fmri/results/l2/zscore_sem/';
feats_dir = '/usr1/afyshe/fmri/data/20questions/features/';

if ~exist(res_dir,'dir'),
    mkdir(res_dir);
end



feat_types  = {'raw','gp','wmean','wslope','fft_power','fft_phase','cwave_haar'};
%feat_types  = {'raw','wmean','fft_power','fft_phase','cwave_haar'};


numWords=60;
numFolds =30;
num_trials = 5;

for s = 1:length(subj_ids),
    subj = subj_ids{s};
    res_sub_dir = sprintf('%s/%s',res_dir,subj);
    if ~exist(res_sub_dir,'dir'),
        mkdir(res_sub_dir);
    end
    
    %Files have data structures with the following fields
    %'data': The output of the feature transformation
    %'time': The time points for the beginning of each feature window.  This vector will have the same length as the last dimension of 'data'.
    %'window_width': The width of each feature window (i.e. the window for which the given feature was created)
    %'window_step': The step size between adjacent windows.  If window_step = window_width then the windows do not overlap at all.
    %'words': the order of the words as they appear in 'data'.
    
    all_data = single([]);
    combo_str = '';
    for f_type = 1:length(feat_types),
        
        
        cur_fname = sprintf('%s/%s/%s_%s_avrg.mat',feats_dir,subj,subj,feat_types{f_type});
        if exist(cur_fname,'file') == 0,
            fprintf('Warning - file does not exist: %s\n',cur_fname);
            continue;
        end
        
        fprintf('Now load %s\n',cur_fname);
        struct = load(cur_fname);
        
        time = struct.time;
        
        %         t_ind = time >=-10 & time <=10;
        %         time = struct.time(t_ind);
        %         struct.data = struct.data(:,:,t_ind);
        
        t_ind = time >=time_start & time <=time_end;
        time = struct.time(t_ind);
        if ndims(struct.data) == 3,
            struct.data = struct.data(:,:,t_ind);
        else
            if ndims(struct.data) ~= 4,
                fprintf('Unknown num dims %i\n', ndims(struct.data) );
                die;
            end
            struct.data = struct.data(:,:,:,t_ind);
        end
        
        words = struct.words;
        s_size= size(struct.data);
        num_sensors = prod(s_size(2:end-1));
        time_length = length(time);
        cur_data = single(reshape(struct.data,[s_size(1),num_sensors*time_length]));
        all_data = cat(2,all_data,cur_data);
        combo_str = sprintf('%s_%s',combo_str,feat_types{f_type});
    end
    
    clear cur_data struct;
    
    for tr = 1:num_trials,
        fprintf('Trial %i, %s %s\n',tr,combo_str,datestr(now));
        %rng(tr);
        rs = RandStream('mcg16807', 'Seed',tr);
        RandStream.setDefaultStream(rs);
        folds = crossvalind('Kfold',numWords,numFolds);
        doOneCrossValNoZYesBias(all_data,sem_matrix, folds, numFolds, numWords, sprintf('%s/%s%s_%i.mat',res_sub_dir,subj,combo_str,tr),0);
        
    end
    
    
end
