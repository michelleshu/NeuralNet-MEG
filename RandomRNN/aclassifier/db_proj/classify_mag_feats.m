function classify_mag_feats(K, R, TS)

load('./sem_matrix.mat');
zscore_w1000 = 0;
zscore_data = 0;

numWords=60;
numFolds =30;
num_trials = 5;
zscore_folds =0;
time_start=0;
time_end=0.75;

if zscore_w1000 == 0,
    sem_matrix = zscore(sem_matrix);
else
    i1000 = load('./bagOfFeatures.mat');
    
    sem_mat1000 = (i1000.features(:,1:218)-3)/2;
    
    sem_mat1000(1:60,:) = sem_matrix;
    
    sem_mat1000 = zscore(sem_mat1000);
    sem_matrix = sem_mat1000(1:60,:);
end



sem_length = size(sem_matrix,2);

subj_ids = {'A'};

% feats_dir = '~/fmri/data/20questions/features/';
% res_dir = '/other/bdstore01x/afyshe/fmri/results/20questions/l2/z_data_zscore1000/';

%subj_ids = {'F','G','I','J'};%{'A','B','C','D','E','F','G','I','J'};
%res_dir = '~/research/fmri/results/l2/zscore_sem_infold_0-0.75/';
%feats_dir = '~/research/fmri/data/20questions/features/';


res_dir = './results';
feats_dir = './data';

% subj_ids = {'A','B','C'};%{'A','B','C','D','E','F','G','I','J'};
% res_dir = '~/research/fmri/results/l2/noz_data/';
% feats_dir = '~/research/fmri/data/20questions/features/';

if ~exist(res_dir,'dir'),
    mkdir(res_dir);
end


feat_str = sprintf('crnn_%dK_%dR_%dTS', K, R, TS);
feat_types  = {feat_str};%'raw','gp','wmean','wslope'};%,'fft_power','fft_phase'};%,'cwave_haar'};%,'corr','cwave_haar'};

save(sprintf('%sparams.mat',res_dir),'numWords','numFolds','num_trials',...
    'zscore_folds','time_start','time_end','sem_matrix','zscore_w1000',...
    'zscore_data');

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
    
    
    for f_type = 1:length(feat_types),
        
        
        cur_fname = sprintf('%s/%s/%s_%s_avrg.mat',feats_dir,subj,subj,feat_types{f_type});
        if exist(cur_fname,'file') == 0,
            fprintf('Warning - file does not exist: %s\n',cur_fname);
            continue;
        end
        
        fprintf('Now load %s\n',cur_fname);
        struct = load(cur_fname);
        
       % COMMENTED OUT MS
%         time = struct.time;
%         
%         t_ind = time >=time_start & time <=time_end;
%         time = struct.time(t_ind);
%         if ndims(struct.data) == 3,
%             struct.data = struct.data(:,:,t_ind);
%         else
%             if ndims(struct.data) ~= 4,
%                 fprintf('Unknown num dims %i\n', ndims(struct.data) );
%                 die;
%             end
%             struct.data = struct.data(:,:,:,t_ind);
%         end
        
        words = struct.words;
        s_size= size(struct.data);
%        num_sensors = prod(s_size(2:end-1));
%        time_length = length(time);
%        struct.data = reshape(struct.data,[s_size(1),num_sensors,time_length]);
        if zscore_data ==1,
            struct.data = zscore(struct.data);
        end
        %cur_data = double(cur_data);
        
        for tr = 1:num_trials,
            fprintf('Trial %i, %s\n',tr,datestr(now));
            rs = RandStream('mcg16807', 'Seed',tr);
            RandStream.setGlobalStream(rs);
            folds = crossvalind('Kfold',numWords,numFolds);
%             doOneCrossValNoZ(struct.data,sem_matrix, folds, numFolds, numWords, ...
%                 sprintf('%s/%s_%s_%i.mat',res_sub_dir,subj,...
%                 feat_types{f_type},tr),zscore_folds);
            
            
            if strcmp('cwave_haar',feat_types{f_type}),
                doOneCrossValNoZ(struct.data,sem_matrix(:,1:100), folds, numFolds, numWords, ...
                    sprintf('%s/%s_%s_%i_1-100.mat',res_sub_dir,subj,...
                    feat_types{f_type},tr),zscore_folds);
                doOneCrossValNoZ(struct.data,sem_matrix(:,101:end), folds, numFolds, numWords, ...
                    sprintf('%s/%s_%s_%i_101-end.mat',res_sub_dir,subj,...
                    feat_types{f_type},tr),zscore_folds);
            else
                % disp(size(struct.data));
                doOneCrossValNoZ(struct.data, sem_matrix, folds, numFolds, numWords, ...
                    sprintf('%s/%s_%s_%i.mat',res_sub_dir,subj,...
                    feat_types{f_type},tr),zscore_folds, 'cosine');
            end
            
        end
        
    end
    
end
end