function classifyMagFeats(subject, dataFile, resDir, sem_matrix)
% Input Parameters:
% subject - one letter identifying which subject this is
% dataFile - filepath where input data can be found
% resDir - main directory where classifier results are stored;
%           subdirectories created for each subject
% sem_matrix - matrix of semantic features

zscore_w1000 = 0;
zscore_data = 0;

numWords=60;
numFolds =30;
num_trials = 5;
zscore_folds = 0;

sem_matrix = zscore(sem_matrix);

save(sprintf('%sparams.mat',resDir),'numWords','numFolds','num_trials',...
    'zscore_folds','sem_matrix','zscore_w1000',...
    'zscore_data');


% Files have data structures with the following fields
% 'data': The output of the feature transformation
% 'time': The time points for the beginning of each feature window.  
%   This vector will have the same length as the last dimension of 'data'.
% 'words': the order of the words as they appear in 'data'.

if exist(dataFile,'file') == 0,
    fprintf('Warning - file does not exist: %s\n',dataFile);
    return;
end

res_sub_dir = sprintf('%s/%s', resDir, subject);
if ~exist(res_sub_dir, 'dir')
    mkdir(res_sub_dir);
end

fprintf('Now load %s\n', dataFile);
struct = load(dataFile);

if zscore_data ==1,
    struct.data = zscore(struct.data);
end

for tr = 1:num_trials,
    fprintf('Trial %i, %s\n',tr,datestr(now));
    rs = RandStream('mcg16807', 'Seed',tr);
    RandStream.setGlobalStream(rs);
    folds = crossvalind('Kfold',numWords,numFolds);
    doOneCrossValNoZ(struct.data, sem_matrix, folds, numFolds, numWords, ...
        sprintf('%s/%s_sparse_%i.mat', res_sub_dir, subject, tr), ...
        zscore_folds, 'cosine');
end

end