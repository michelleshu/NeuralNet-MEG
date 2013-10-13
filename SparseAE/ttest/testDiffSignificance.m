subjects = {'B', 'C', 'D', 'E', 'F', 'G', 'I', 'J'};

p_values = zeros(8, 1);

for s = 1 : numel(subjects)
    subject = subjects{s};
    
    raw_dists = zeros(300, 1);
    sae_dists = zeros(300, 1);
    
    r = 1;  % row of dists vectors to write to
    for trial = 1 : 5
        raw_file = ...
            sprintf('./results/classify_raw/%s/%s_raw_avrg_%i.mat', ...
            subject, subject, trial);
        
        sae_file = ...
            sprintf('./results/classify_sae/%s/%s_sparse_1.000e-01R_100K_%i.mat', ...
            subject, subject, trial);
    
        % For each trial, we will retrieve the 60 distances between raw MEG
        % predictions and true values and the 60 distances between sparse AE
        % predictions and true values.
        [raw1, raw2] = getDistToCorrect(raw_file);
        [sae1, sae2] = getDistToCorrect(sae_file);
        
        raw_dists(r : r + 29, 1) = raw1;
        raw_dists(r + 30 : r + 59, 1) = raw2;
        sae_dists(r : r + 29, 1) = sae1;
        sae_dists(r + 30 : r + 59, 1) = sae2;
        
        r = r + 60;
    end
    
    diff = raw_dists - sae_dists;
    %[h, p] = ttest(diff, 0, 'Alpha', 0.05, 'Tail', 'left');
    [h, p] = ttest2(raw_dists, sae_dists, 'Alpha', 0.05, 'Tail', 'right');
    %[p,h] = ranksum(raw_dists, sae_dists, 'Alpha', 0.05, 'Tail', 'right');
    p_values(s) = p;
    fprintf('Subject %s: \n', subject);
    fprintf('h = %i\n', h);
    fprintf('p = %d\n\n', p); 
end

chi_p = 1-chi2cdf(-2*sum(log(p_values)),length(p_values)*2);
disp(chi_p);