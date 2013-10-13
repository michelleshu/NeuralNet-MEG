% Load stuff

InitParam = readInitParam('InitParam.txt');
[tmp,cx,cy,cl] = textread(InitParam.sensorfile,'%f %f %f %s');
shift_cx = round((cx - min(cx))*140);
shift_cy = round((cy - min(cy))*18);



%br = load('BrainRegions');
%br.Parietal =  setdiff(br.Parietal, [184 185 186]);
%br.Visual =  sort([[184 185 186]' ; br.Visual]);


sensors = 2:3:306;

b_vis = br.Visual(ismember(br.Visual,sensors));
b_front = br.Frontal(ismember(br.Frontal,sensors));
b_temp = br.Temporal(ismember(br.Temporal,sensors));
b_pari = br.Parietal(ismember(br.Parietal,sensors));


%data = randn(306,150);
% grad 1 1:3:306 grad 2 2:3:306 mags (3:3:306)
% data = data(1:3:306,:);

sens_index = 1:3:306;

% this is for the 204 sensors I used to make the pattern plots
% sens_306_space = sort([1:3:306 2:3:306]);
% 
% sens = sort([1:3:306 2:3:306]);
% 
% sens_frontal = find(ismember(sens, br.Frontal));
% sens_visual = find(ismember(sens, br.Visual));
% sens_parietal = find(ismember(sens, br.Parietal));
% sens_temporal = find(ismember(sens, br.Temporal));
% 
% sens =  [sens_frontal sens_temporal sens_parietal];%  sens_visual ];
% sens_index = sens_306_space(sens(1:2:end));

% scatter(shift_cx(b_vis),shift_cy(b_vis),r2(b_vis)*10^3,r2(b_vis)*10,'o','filled'); axis off;
% hold all
% scatter(shift_cx(b_front),shift_cy(b_front),r2(b_front)*10^3,r2(b_front)*10,'<','filled'); axis off;
% scatter(shift_cx(b_temp),shift_cy(b_temp),r2(b_temp)*10^3,r2(b_temp)*10,'>','filled'); axis off;
% scatter(shift_cx(b_pari),shift_cy(b_pari),r2(b_pari)*10^3,r2(b_pari)*10,'s','filled'); axis off;



%%

subj_num=8;
square_size = 450;
fontsize = 18;

pa = data;
%(squeeze(all_pats(subj_num,[sens(1:2:end) sens(2:2:end)],:,i)));%+abs(reshape(all_pats(subj_num,2:2:end,:,i),102,[]));
c_max = max(abs(pa(:)));
c_min = 0;%-1*c_max;
corr_mat_mean = pa;
%squeeze(mean(corr_mat_kick_a(:,(b_pari-2)/3+1,:),2));
time= 0:0.005:0.5;
nframes=length(time);
M=moviein(nframes);
dt=1;

for it=1:nframes
    subplot(5,1,1:4);
    cla
    
    axis off;
    hold on;
    % these two ones are to set the max and the min for the color scale
    scatter(shift_cx(1),shift_cy(1),...
        square_size-10,c_max,'o','filled');
    scatter(shift_cx(1),shift_cy(1),...
        square_size-10,c_min,'o','filled');
    
    %         scatter(shift_cx(sensors),shift_cy(sensors),...
    %             (ones(size(sensors)))*square_size,...
    %             (corr_mat_mean(:,it)),'s','filled');
    
    % here we are plotting the actual data
    scatter(shift_cx(sens_index),shift_cy(sens_index),...
        (ones(1,length(sens_index)))*square_size,...
        (corr_mat_mean(:,it)),'o','filled');
    
    %         scatter([shift_cx(sens_index) ;shift_cx(sens_index)+400],...
    %             [shift_cy(sens_index) ;shift_cy(sens_index)],...
    %             (ones(1,length(sens_index)*2))*square_size,...
    %             (corr_mat_mean(:,it)),'o','filled');
    
    title(sprintf('Subject A'),'fontsize',fontsize)
    % title('Pattern matrix for word ');
    
    c=colorbar ('FontSize',12);
    caxis([c_min, c_max]);
    ylabel(c,'Pattern Score')
    
    hold off;
    
    subplot(5,1,5);
    cla
    hold on
    %h = plot(time, [zeros(1, dt*it) ones(1, length(time) - dt*it)],'r');
    %set(h,'LineWidth',2);
    axis([time(1) time(end) 0 1]);
    if time(it) < 0.8,
        line([time(it) time(it)],[0 1],'color','red','linewidth',2)
    else
        line([time(it) time(it)],[0 1],'color','green','linewidth',2)
    end
    set(gca,'YTick',[]);
    title(sprintf('t = %.f ms',1000*time(it*dt)),'fontsize',fontsize);
    hold off
    
    drawnow;
    
    
    pos_vec = get(gcf,'Position');
    M(:,it)=getframe(gcf, [0 0 pos_vec(3:4)]);
    hold off;
    pause;
end

%movie2avi(M, sprintf('/Users/afyshe/pattern_movie_square_sum_grad_subj%s_%s.avi',...
%     subj_chars{subj_num},all_words{adj_inds(9-1*(size(p,2)-i))}),...
%     'compression','none');

%waitforbuttonpress


