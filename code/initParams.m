function params = initParams()

% load the data to be used
load('/Users/michelleshu/Documents/Mitchell/CRNN-MEG/data/D/D_raw_avrg.mat');
params.data = data(:, :, 53:202);   % cut to range 0 - 750 ms
params.time = time(53:202);
params.wordNames = words;

% load the semantic features matrix
load('/Users/michelleshu/Documents/Mitchell/CRNN-MEG/sem_matrix2.mat');
params.semMatrix = sem_matrix;

% set the number of words that we have data for
params.numWords = 60;

% set the number of time points in our timeseries
params.numTimePoints = 150;

% set the number of sensors
params.numSensors = 306;

% set the number of first layer CNN filters
params.numFilters = 50;

% load the lists of sensors in each brain region
load('/Users/michelleshu/Documents/Mitchell/CRNN-MEG/brainregions/brainregions.mat');
params.brainRegions = brainRegions;

% save number of brain regions
params.numRegions = numel(fieldnames(params.brainRegions));

% set the number of RNN to use
params.numRNN = 64;
