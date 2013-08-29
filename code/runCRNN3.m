train_iters = 5;
subject_ids = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'J'};
results_dir = './results';
K_options = [90, 100];
R_options = [60, 80, 100, 120, 140, 160, 180];

for k_index = 1 : numel(K_options)
    k = K_options(k_index); % select number of filters to train
    
    for r_index = 1 : numel(R_options)
        r = R_options(r_index); % select number of RNNs to train

        one_mean_acc = zeros(numel(subject_ids), train_iters);
        two_mean_acc = zeros(numel(subject_ids), train_iters);

        for i = 1 : train_iters
            disp(['Iteration: ' num2str(i)]);
            disp(['K = ' num2str(k) '    R = ' num2str(r)]);

            % Train RNN for each subject's data
            for s = 1 : length(subject_ids)
                rng('shuffle');
                subject = subject_ids{s};
                disp(['Training Subject ' subject]);
                [filters, params] = pretrain(initParams(subject, k, r));
                forwardRNN(filters, params);
            end

            % Run all RNN word representations through classifier.
            classify_mag_feats;

            % Analyze accuracy of classification. Print results to console.
            for s = 1 : length(subject_ids)
                subject = subject_ids{s};
                acc_ones = zeros(5, 1);
                acc_twos = zeros(5, 1);
                for t = 1 : 5
                    filename = sprintf('%s/%s/%s_crnn_%s.mat', results_dir, ...
                        subject, subject, num2str(t));
                    [acc_ones(t), acc_twos(t)] = getAccuracy(filename);
                end

                one_mean_acc(s, i) = mean(acc_ones);
                two_mean_acc(s, i) = mean(acc_twos);
            end
        end
        
        save(sprintf('./acc_results/acc_%dK_%dR.mat', k, r), ...
            'one_mean_acc', 'two_mean_acc');
        fprintf('Accuracy results saved to ./acc_results/acc_%dK_%dR.mat', k, r);
    end
end
