%% Scale up sparse representation data to a larger range to enhance
%% classification accuracy

% subjects = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'J'};
% dataDir = './results/sparse';
% fileSuffix = '1.000e-01R_100K.mat';
% 
% for s = 1 : numel(subjects)
%     
%     load(sprintf('%s/%s_%s', dataDir, subjects{s}, fileSuffix));
%     
%     % Scale data to range -10 to 10 %
%     dataMin = min(min(min(data)));
%     data = data - dataMin;
% 
%     dataMax = max(max(max(data)));
%     data = data ./ dataMax .* 20;
%     data = data - 10;
%     
%     save(sprintf('%s/%s_upscaled_%s', dataDir, subjects{s}, fileSuffix),...
%         'data', 'time', 'words');
% end

classifyMagFeats('G', './results/sparse/G_upscaled_1.000e-01R_100K.mat', ...
    './results/classify_upscaled', 0.1, ...
    100, './data/sem_matrix.mat');

% -------------------------------------------------------------------------
% Compute classification accuracy
acc_ones = zeros(5, 1);  % 1 v 2 acc over 5 trials
acc_twos = zeros(5, 1);  % 2 v 2 acc over 5 trials
for trial = 1 : 5
    classifyFile = ...
        sprintf('./results/classify_upscaled/G/G_sparse_1.000e-01R_100K_%i.mat', trial);
    [acc_ones(trial), acc_twos(trial)] = getAccuracy(classifyFile);
end
fprintf('1 v 2 Accuracy: %2.3f\n', mean(acc_ones));
fprintf('2 v 2 Accuracy: %2.3f\n', mean(acc_twos));
