% Train the baseline classifiers
numWords=60;
numFolds =30;

base_dir = '/usr1/afyshe/fmri/dbproj/';
folds = crossvalind('Kfold',numWords,numFolds);
load ~/sem_matrix.mat


subj_ids = {'A','B','C','D','E','F','G','I','J'};
for j = 1:length(subj_ids),
    subj = subj_ids{j};
    
    d20q = load(sprintf('/usr1/meg/20questions/decoding/%s/%s_sensors_SSSt_SSP_LP50_DS200_tc_noBlinksSSP.mat',subj, subj));
    d20q.data = permute(d20q.data, [2 1 3])*10^12;
    avrg_win = d20q.time >=-0.2 & d20q.time <=0;
    use_win = d20q.time >=0 & d20q.time <=1;
    
    my_data = double(d20q.data - repmat(mean(d20q.data(:,:,avrg_win),3), [1,1,340]));
    my_data = my_data(:,:,use_win);
    
    mag_data = zeros(60,size(my_data,2),size(my_data,3));
    for w = 1:60,
        mag_data(w,:,:) = squeeze(mean(my_data(d20q.labels==w,:,:)));
    end
    
    
    cur_data = zscore(mag_data(:,:));
    
    preds = zeros(60);
    ranks = zeros(1,60);
    ests = zeros(60,218);
    for i = 1:numFolds,
        fprintf('Fold %i... ',i);
        [weightMatrix,r]=learn_text_from_fmri_kernel(cur_data(folds~=i,:),...
            sem_matrix(folds~=i,:),1);
        
        testBrainExample=cur_data(folds==i,:);
        testWordNums = find(folds==i);
        % append a vector of ones for the biases
        testBrainExample(:,end+1) = 1;
        
        for cur_word = 1:size(testBrainExample,1),
            
            targetEstimate = testBrainExample(cur_word,:)*weightMatrix;
            ests(testWordNums(cur_word),:) = targetEstimate;
            d = pdist([targetEstimate; sem_matrix], 'euclidean');
            [y,ind] = sort(d(1:numWords));
            ranks(testWordNums(cur_word)) = find(testWordNums(cur_word)==ind);
            preds(testWordNums(cur_word),:) = ind;
        end
        
    end
    fprintf('\n');
    save(sprintf('%s/%s_mag_class_results_t0-1_zsem.mat',base_dir,subj),'ests','ranks','preds','folds');
end



