
% I didn't end up using the PLV features, but here's how I made them.
br = load('~/research/fmri/BrainRegions.mat');
subj_ids = {'A','D'};
load ~/sem_matrix.mat

num_words=60;

br_fields = fields(br);

f_thresh = 60;
freq=200;
window_size=200;
window_overlp = 25;


base_dir = '/usr1/afyshe/fmri/dbproj/';

for j = 1:length(subj_ids),
    subj = subj_ids{j};
    
    d20q = load(sprintf('/usr1/meg/20questions/decoding/%s/%s_sensors_SSSt_SSP_LP50_DS200_tc_noBlinksSSP.mat',subj, subj));
    t_window = d20q.time >=0 & d20q.time <=1;
    
    d20q.data = permute(d20q.data, [2 1 3])*10^12;
    avrg_win = d20q.time >=-0.2 & d20q.time <=0;
    my_data = double(d20q.data - repmat(mean(d20q.data(:,:,avrg_win),3), [1,1,340]));
    my_data = my_data(:,2:3:306,t_window);
    num_winds = length(1:window_overlp:size(my_data,3)-window_size+1);
    
    plv_size = size(my_data,2)*(size(my_data,2)-1)/2;
    plv_window = zeros([num_words, plv_size, f_thresh, num_winds],'single');
    fprintf('Start plv %s %s\n',subj,datestr(now));
    
    for i = 1:num_words,
        if rem(i,2) == 1,
            fprintf('%s word %i... \n',datestr(now),i);
        end
        
        cur_data = my_data(d20q.labels==i,:,:);
        cs = size(cur_data);
        cur_data = reshape(detrend(cur_data(:,:)')',cs);
        for c = 1:size(cur_data,1),
            if rem(c,5) == 1,
                fprintf('%s trial %i... ',datestr(now),c);
            end
            plv_ind = 1;
            for sens1 = 1:size(cur_data,2),
                data1 = detrend(squeeze(cur_data(c,sens1,:)));
                [p,~,~]=traces2PLF_nodetrend(data1,1:f_thresh,freq,7);
                
                for sens2 = sens1+1:size(cur_data,2),
                    
                    data2 = squeeze(cur_data(c,sens2,:));
                    [p2,~,~]=traces2PLF_nodetrend(data2,1:f_thresh,freq,7);
                    plv = exp(1i*angle(p.*conj(p2)));
                    
                    plv_w = zeros(f_thresh,num_winds,'single');
                    pind=1;
                    for w = 1:window_overlp:size(cur_data,3)-window_size+1,
                        plv_w(:,pind) = abs(mean(plv(:,w:w+window_size-1),2));
                        pind = pind+1;
                    end
                    
                    plv_window(i,plv_ind,:,:) = squeeze(plv_window(i,plv_ind,:,:)) + plv_w;
                    plv_ind = plv_ind+1;
                    
                    
                end
            end
        end
        
        plv_window(i,:,:,:) = plv_window(i,:,:,:)/size(cur_data,1);
    end
    fprintf('\n');
    save(sprintf('%s%s_plv_feats.mat',base_dir,subj),'plv_window','-v7.3');
    clear plv_window;
    
end

