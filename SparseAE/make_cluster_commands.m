subj_chars= {'B', 'C', 'D', 'E', 'F', 'G', 'I', 'J'};
rhos = 0.05;
ks = 200;
code_dir = '~/code/';
results_dir='~/results/';
data_dir='~/data/meg/20questions/features/';
sem_mat_file='~/sem_matrix.mat';

for i =1:length(subj_chars),
    for r = 1:length(rhos),
        for k = 1:length(ks);
fprintf('./singlenode.pl sae_%s_%.3f_%i "./matlab_batcher.sh \\" addpath(genpath(''%s'')); trainSAE(%.3f,%i,''%s'',''%s'',''%s'',''%s'');\\" \\"sae_%s_%.3f_%i\\""\n',...
    subj_chars{i},rhos(r),ks(k),...
    code_dir,...
    rhos(r),ks(k),subj_chars{i},results_dir,data_dir,sem_mat_file,... % params
    subj_chars{i},rhos(r),ks(k));
        end
    end
end