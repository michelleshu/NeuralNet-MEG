addpath classifier

subjects = {'A'};
R_options = [0.01, 0.05, 0.1, 0.2];
K_options = [100, 200, 300];
acc_ones = zeros(numel(K_options), numel(R_options));
acc_twos = zeros(numel(K_options), numel(R_options));

for s = 1 : numel(subjects)
    prefix = sprintf('%s/%s_sparse', subjects{s}, subjects{s});
    for r_index = 1 : numel(R_options)
        R = R_options(r_index);

        for k_index = 1 : numel(K_options)
            K = K_options(k_index);

            trial_accs_ones = zeros(5, 1);
            trial_accs_twos = zeros(5, 1);

            for trial = 1 : 5
                [trial_accs_ones(trial), trial_accs_twos(trial)] = ...
                    getAccuracy(sprintf('new_results/classify/%s_%1.3dR_%iK_%i.mat', ...
                    prefix, R, K, trial));
            end

            acc_ones(k_index, r_index) = mean(trial_accs_ones);
            acc_twos(k_index, r_index) = mean(trial_accs_twos);
        end
    end
    fprintf('\nSubject %s\n', subjects{s});
    fprintf('%2.3f\n%2.3f\n', mean(trial_accs_ones), mean(trial_accs_twos));
end