%function makeHelmetMovie% (subject, featureIndex)

% Load pre-computed network weights
% load(sprintf('./code/visualization/weights/%s_one_feat_weights_z.mat', subject));

%numComponents = 30;
timeSize = 150;

%optInputs = getOptHiddenNodeInputs(W1);
%origInputs = reversePCAInput(subject, optInputs, numComponents, timeSize);
%data = getWeightedAverage(W2, origInputs, featureIndex);

% Load stuff
load('./code/visualization/fullBrainRegions.mat');
InitParam = readInitParam('InitParam.txt');
[tmp,cx,cy,cl] = textread(InitParam.sensorfile,'%f %f %f %s');
shift_cx = round((cx - min(cx))*140);
shift_cy = round((cy - min(cy))*18);

%data = squeeze(inputs(hiddenIndex, :, :));
data1 = data(1:3:306, :);
data2 = data(2:3:306, :);
data3 = data(3:3:306, :);

sensors = 1:3:306;

b_vis = br.Visual(ismember(br.Visual,sensors));
b_front = br.Frontal(ismember(br.Frontal,sensors));
b_temp = br.Temporal(ismember(br.Temporal,sensors));
b_pari = br.Parietal(ismember(br.Parietal,sensors));


sens_index = 1:3:306;
square_size = 450;
fontsize = 18;

c_max = max([max(abs(data1(:))), max(abs(data2(:))), max(abs(data3(:)))]);
c_min = 0;%-1*c_max;

nframes=timeSize;
M=moviein(nframes);
figure('units','normalized','position',[0 1 1 .4]);

for it=1:nframes 
    axis off;
    hold on;
    
    subplot(1, 3, 1);
    % these two ones are to set the max and the min for the color scale
    scatter(shift_cx(1),shift_cy(1),...
        square_size-10,c_max,'o','filled');
    scatter(shift_cx(1),shift_cy(1),...
        square_size-10,c_min,'o','filled');
    
    % here we are plotting the actual data
    scatter(shift_cx(sens_index),shift_cy(sens_index),...
        (ones(1,length(sens_index)))*square_size,...
        (data1(:,it)),'o','filled');
    axis off;
    
    subplot(1, 3, 2);
    scatter(shift_cx(1),shift_cy(1),...
        square_size-10,c_max,'o','filled');
    scatter(shift_cx(1),shift_cy(1),...
        square_size-10,c_min,'o','filled');
    scatter(shift_cx(sens_index),shift_cy(sens_index),...
        (ones(1,length(sens_index)))*square_size,...
        (data2(:,it)),'o','filled');
    time = (it - 1) * 750 / timeSize;
    title(sprintf('Time = %i ms', time), 'FontSize', 18);
    axis off;
    
    subplot(1, 3, 3);
    scatter(shift_cx(1),shift_cy(1),...
        square_size-10,c_max,'o','filled');
    scatter(shift_cx(1),shift_cy(1),...
        square_size-10,c_min,'o','filled');
    scatter(shift_cx(sens_index),shift_cy(sens_index),...
        (ones(1,length(sens_index)))*square_size,...
        (data3(:,it)),'o','filled');
    axis off;

    %c=colorbar ('FontSize',12);
    %caxis([c_min, c_max]);
    
    %time = (it - 1) * 750 / timeSize;
    %title(sprintf('Time = %i ms', time), 'FontSize', 18);
    drawnow;
   
    %pos_vec = get(gcf,'Position');
    %M(:,it)=getframe(gcf, [0 0 pos_vec(3:4)]);
    M(:,it)=getframe(gcf);
    hold off;
end

movie2avi(M, './code/visualization/movies/SAE_alive.avi', 'fps', 1);