addpath classifier
diary on
diary('./output.txt');
subjects = {'A', 'B', 'C' 'D', 'E', 'F', 'G', 'I', 'J'};
for i = 1 : numel(subjects)
    subject = subjects{i};
    classifyMagFeatsRaw(subject, ...
        sprintf('./data/%s_raw_avrg.mat', subject), ...
        './results/classify',  './data/sem_matrix.mat');
    % Compute classification accuracy
    acc_ones = zeros(5, 1);  % 1 v 2 acc over 5 trials
    acc_twos = zeros(5, 1);  % 2 v 2 acc over 5 trials
    for trial = 1 : 5
        classifyFile = ...
            sprintf('./results/classify/%s/%s_raw_avrg_%i.mat', ...
                subject, subject, trial);
        [acc_ones(trial), acc_twos(trial)] = getAccuracy(classifyFile);
    end
    fprintf('1 v 2 Accuracy: %2.3f\n', mean(acc_ones));
    fprintf('2 v 2 Accuracy: %2.3f\n', mean(acc_twos));
end