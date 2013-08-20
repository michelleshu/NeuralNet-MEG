
%Calculate results based on classification output.

numFolds = 30;
load('~/sem_matrix')

afolds = load('~/school/databases/project/A_class_folds.mat');

subj_ids = {'A','B','C','D','E','F','G','I','J'};

file_names = {
    '~/school/databases/project/%s_mag_class_results_t0-1.mat',...
    '~/school/databases/project/%s_harr_descr_class_results.mat',...
    '~/school/databases/project/%s_harr_wave_class_results.mat',...
    '~/school/databases/project/%s_morl_wave_class_results.mat',...
    '~/school/databases/project/%s_spec_ang_class_results.mat',...
    '~/school/databases/project/%s_spec_class_results.mat',...
    '~/school/databases/project/%s_spec_mag_class_results.mat'};

technique_names = {'Magnitude',...
    'Discrete Haar',...
    'Continuous Haar',...
    'Continuous Morlet',...
    'STFT (Phase)',...
    'STFT (Power \& Phase)',...
    'STFT (Power)'};

% table_str is just to make the results table for the report
table_str = '';
avrg_res = zeros(2,length(file_names));

for fnum = 1:length(file_names),
    cur_file_pref = file_names{fnum};
     table_str = sprintf('%s %s & ',table_str, technique_names{fnum});
    for j = 1:length(subj_ids),
        subj = subj_ids{j};
        resw = load(sprintf(cur_file_pref,subj));
         if isfield(resw,'folds') == 1,
             folds = resw.folds;
         else
           folds = afolds.folds; 
         end
        preds = resw.ests;
        
        twos = zeros(1,30);
        wons = zeros(2,30);
        boths = zeros(1,30);
        dists = zeros(2,2,30);
        
        for i = 1:numFolds,
            
            testWordNums = find(folds==i);
            
            if length(testWordNums) ~= 2,
                die ;
            end
            
            cur_preds = preds(testWordNums,:);
            true_labs = sem_matrix(testWordNums,:);
            
            d11 = sqrt(sum((cur_preds(1,:) - true_labs(1,:)) .^ 2));
            d22 = sqrt(sum((cur_preds(2,:) - true_labs(2,:)) .^ 2));
            
            d12 = sqrt(sum((cur_preds(1,:) - true_labs(2,:)) .^ 2));
            d21 = sqrt(sum((cur_preds(2,:) - true_labs(1,:)) .^ 2));
            
            dists(1,1,i)=d11;
            dists(1,2,i)=d12;
            dists(2,1,i)=d21;
            dists(2,2,i)=d22;
            
            if d11 + d22 <= d12 + d21,
                twos(i) = 1;
            end
            if d11 <= d12,
                wons(1,i) = 1;
            end
            
            if d22 <= d21,
                wons(2,i) = 1;
            end
            
       
        end
        fprintf('\n********\nSubj %s\n2x2 %.4f\n1v2 %.4f\nMean rank:%.2f\nMedian rank:%.2f\n********\n',...
            subj,mean(twos(:)),mean(wons(:)), mean(resw.ranks), median(resw.ranks));
        table_str = sprintf('%s %.4f & %.4f & ',table_str,mean(twos(:)),mean(wons(:)));
        avrg_res(1,fnum) = avrg_res(1,fnum) + mean(twos(:));
        avrg_res(2,fnum) = avrg_res(2,fnum) + mean(wons(:));
    end
    table_str = sprintf('%s\\\\ \n',table_str);
end


