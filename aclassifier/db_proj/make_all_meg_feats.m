
br = load('~/research/fmri/BrainRegions.mat');
subj_ids = {'A','B','C','D','E','F','G','I','J'};
load ~/sem_matrix.mat

num_words=60;

br_fields = fields(br);

f_thresh = 60;
freq=200;
scales = 1:64;

base_dir = '/usr1/afyshe/fmri/dbproj/';

for j = 1:length(subj_ids),
    subj = subj_ids{j};
    
    d20q = load(sprintf('/usr1/meg/20questions/decoding/%s/%s_sensors_SSSt_SSP_LP50_DS200_tc_noBlinksSSP.mat',subj, subj));
    d20q.data = permute(d20q.data, [2 1 3])*10^12;
    avrg_win = d20q.time >=-0.2 & d20q.time <=0;
    my_data = double(d20q.data - repmat(mean(d20q.data(:,:,avrg_win),3), [1,1,340]));
    
    
    
    
    fprintf('Start wave %s %s\n',subj,datestr(now));
    [c,l] = wavedec(my_data(1,1,:),7,'haar');
    w_coeff = zeros([num_words, size(my_data,2), length(c)],'single');
    
    for i = 1:num_words,
        if rem(i,10) == 1,
            fprintf('word %i... ',i);
        end
        
        cur_data = my_data(d20q.labels==i,:,:);
        for c = 1:size(cur_data,1),
            for sens = 1:size(cur_data,2),
                [coeff,l]=wavedec(squeeze(cur_data(c,sens,:)),7,'haar');
                w_coeff(i,sens,:,:) = squeeze(w_coeff(i,sens,:)) + coeff;
            end
        end
        w_coeff(i,:,:,:) = w_coeff(i,:,:,:)/size(cur_data,1);
    end
    fprintf('\n');
    save(sprintf('%s%s_harr_descr_feats.mat',base_dir,subj),'w_coeff','-v7.3');
    
    
    [spec1,f1,t1,p1]=spectrogram(squeeze(my_data(1,1,:)),20,10,[],freq);
    
    f_cut = find(f1>f_thresh);
    
    spec_ang = zeros([num_words, size(my_data,2), size(spec1(1:f_cut,:))],'single');
    spec_mag = zeros([num_words, size(my_data,2), size(spec1(1:f_cut,:))],'single');
    fprintf('Start spec %s %s\n',subj,datestr(now));
    
    for i = 1:num_words,
        if rem(i,10) == 1,
            fprintf('word %i... ',i);
        end
        
        cur_data = my_data(d20q.labels==i,:,:);
        for c = 1:size(cur_data,1),
            for sens = 1:size(cur_data,2),
                [spec1,f1,t1,p1]=spectrogram(squeeze(cur_data(c,sens,:,:)),20,10,[],freq);
                spec_ang(i,sens,:,:) = squeeze(spec_ang(i,sens,:,:)) + angle(spec1(1:f_cut,:));
                spec_mag(i,sens,:,:) = squeeze(spec_mag(i,sens,:,:)) + abs(spec1(1:f_cut,:));
            end
        end
        spec_ang(i,:,:,:) = spec_ang(i,:,:,:)/size(cur_data,1);
        spec_mag(i,:,:,:) = spec_mag(i,:,:,:)/size(cur_data,1);
    end
    fprintf('\n');
    save(sprintf('%s%s_spec_feats.mat',base_dir,subj),'spec_mag','spec_ang','-v7.3');
    clear spec_mag spec_ang;
    
    % wavelets
    cwave1_morl = cwt(squeeze(my_data(1,1,:)),scales,'morl');
    cwave_morl = zeros([num_words, size(my_data,2), size(cwave1_morl)],'single');
    cwave_haar = zeros([num_words, size(my_data,2), size(cwave1_morl)],'single');
    
    fprintf('Start wave %s %s\n',subj,datestr(now));
    for i = 1:num_words,
        if rem(i,10) == 1,
            fprintf('word %i... ',i);
        end
        cur_data = my_data(d20q.labels==i,:,:);
        for c = 1:size(cur_data,1),
            for sens = 1:size(cur_data,2),
                c_data = squeeze(cur_data(c,sens,:));
                cwave_morl(i,sens,:,:) = squeeze(cwave_morl(i,sens,:,:)) + cwt(c_data,scales,'morl');
                cwave_haar(i,sens,:,:) = squeeze(cwave_haar(i,sens,:,:))+ cwt(c_data,scales,'haar');
            end
        end
        cwave_morl(i,:,:,:) = cwave_morl(i,:,:,:)/size(cur_data,1);
        cwave_haar(i,:,:,:) = cwave_haar(i,:,:,:)/size(cur_data,1);
    end
    fprintf('\n');
    save(sprintf('%s%s_wave_feats.mat',base_dir,subj),'cwave_haar','cwave_morl','-v7.3');
    clear cwave_morl cwave_haar;
    
end

