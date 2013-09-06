train_iters = 1;
subject_ids = {'A'};
results_dir = './results';
K_options = 150;
R_options = 100;
TS_options = 3;

for k_index = 1 : numel(K_options)
    k = K_options(k_index); % select number of filters to train
    
    for r_index = 1 : numel(R_options)
        r = R_options(r_index); % select number of RNNs to train
        
        for ts_index = 1 : numel(TS_options)
            ts = TS_options(ts_index); % select number of time sections

            one_mean_acc = zeros(numel(subject_ids), train_iters);
            two_mean_acc = zeros(numel(subject_ids), train_iters);

            for i = 1 : train_iters
                disp(['Iteration: ' num2str(i)]);
                disp(['K = ' num2str(k) '    R = ' num2str(r) ...
                    '     TS = ' num2str(ts)]);

                % Train RNN for each subject's data
                for s = 1 : length(subject_ids)
                    rng('shuffle');
                    subject = subject_ids{s};
                    disp(['Training Subject ' subject]);
                    [filters, params] = pretrain(initParams(subject, k, r, ts));
                    forwardRNN(filters, params);
                end

                % Run all RNN word representations through classifier.
                classify_mag_feats(k, r, ts);

                % Analyze accuracy of classification. Print results to console.
                for s = 1 : length(subject_ids)
                    subject = subject_ids{s};
                    acc_ones = zeros(5, 1);
                    acc_twos = zeros(5, 1);
                    for t = 1 : 5
                        filename = sprintf('%s/%s/%s_crnn_%dK_%dR_%dTS_%s.mat', results_dir, ...
                            subject, subject, k, r, ts, num2str(t));
                        [acc_ones(t), acc_twos(t)] = getAccuracy(filename);
                        fprintf('Accuracies: %2.3f\t%2.3f \n', ...
                            acc_ones(t), acc_twos(t));  
                    end

                    one_mean_acc(s, i) = mean(acc_ones);
                    two_mean_acc(s, i) = mean(acc_twos);
                end
            end
            
            save(sprintf('./acc_results/acc_%dK_%dR_%dTS.mat', k, r, ts), ...
                'one_mean_acc', 'two_mean_acc');
            fprintf('Accuracy results saved to ./acc_results/acc_%dK_%dR_%dTS.mat\n', ...
                k, r, ts);
        end
    end
end
