function [one_mean_acc, two_mean_acc] = runCRNN()

train_iters = 2;
subject_ids = {'A'}; %, 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'J'};
results_dir = '/Users/michelleshu/Documents/Mitchell/CRNN-MEG/results';

one_mean_acc = zeros(numel(subject_ids), train_iters);
two_mean_acc = zeros(numel(subject_ids), train_iters);

for i = 1 : train_iters
    rng('shuffle');
    disp(['Iteration: ' num2str(i)]);
    
    % Train RNN for each subject's data
    for s = 1 : length(subject_ids)
        subject = subject_ids{s};
        disp(['Training Subject: ' subject]);
        [filters, params] = pretrain(initParams(subject));
        forwardRNN(filters, params);
    end
    
    % Run all RNN word representations through classifier.
    classify_mag_feats;
    
    % Analyze accuracy of classification. Print results to console.
    for s = 1 : length(subject_ids)
        subject = subject_ids{s};
        disp(['Results for Subject ' subject]);
        acc_ones = zeros(5, 1);
        acc_twos = zeros(5, 1);
        for t = 1 : 5
            filename = sprintf('%s/%s/%s_crnn_%s.mat', results_dir, ...
                subject, subject, num2str(t));
            [acc_ones(t), acc_twos(t)] = getAccuracy(filename);
            fprintf('Trial %d\n1 vs 2: %f\t 2 vs 2: %f\n', t, ...
                acc_ones(t), acc_twos(t));
        end
        fprintf('\n');
        fprintf('Av. 1 vs 2: %f\t Av. 2 vs 2: %f\n\n', mean(acc_ones), ...
            mean(acc_twos));
        
        one_mean_acc(s, i) = mean(acc_ones);
        two_mean_acc(s, i) = mean(acc_twos);
    end
end
end