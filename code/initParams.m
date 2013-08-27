function params = initParams(subject, K, R)

% load the data to be used
% load('/Users/michelleshu/Documents/Mitchell/CRNN-MEG/data/F/F_raw_avrg.mat');
data_dir = './data';
params.in_file = sprintf('%s/%s/%s_raw_avrg.mat', data_dir, subject, subject);
params.out_file = sprintf('%s/%s/%s_crnn_avrg.mat', data_dir, subject, subject);
load(params.in_file);
params.data = data(:, :, 53:202);   % cut to range 0 - 750 ms
params.time = time(53:202);
params.wordNames = words;

% load the semantic features matrix
load('./sem_matrix.mat');
params.semMatrix = sem_matrix;

% set the number of words that we have data for
params.numWords = 60;

% set the number of time points in our timeseries
params.numTimePoints = 150;

% set the number of sensors
params.numSensors = 306;

% set the number of first layer CNN filters
params.numFilters = K;

% load the lists of sensors in each brain region
load('./brainregions/brainregions.mat');
params.brainRegions = brainRegions;

% save number of brain regions
params.numRegions = numel(fieldnames(params.brainRegions));

% set the number of RNN to use
params.numRNN = R;
