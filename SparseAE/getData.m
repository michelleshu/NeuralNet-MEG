function [patches, time, words] = getData( dataFile )
%% sampleMEG.m
% Collect all patches of MEG data from the subject for training input
patchWidth = 1; % number of time points to include in a patch
load(dataFile);

% Isolate time range of interest: stimulus onset (0 ms) to 750 ms
data = data(:, :, 53 : 202);
time = time(53 : 202);

% Make 3rd set of sensor values larger for normalization
i = 3;
while i < 306
    data(:, i, :) = squeeze(data(:, i, :)) .* 20;
    i = i + 3;
end

% Normalize data to range 0.1 to 0.9 %
dataMin = min(min(min(data)));
data = data - dataMin;

dataMax = max(max(max(data)));
data = data ./ dataMax .* 0.8;
data = data + 0.1;

numWords = size(data, 1);
numSensors = size(data, 2);
numTimePoints = size(data, 3);
numPatches = numWords * (numTimePoints - patchWidth + 1);

patches = zeros(numSensors * patchWidth, numPatches);

patchIndex = 1;
for word = 1 : numWords
    for timeStart = 1 : numTimePoints - patchWidth + 1
        patch = squeeze(data(word, :, ...
            timeStart : timeStart + patchWidth - 1));
        patches(:, patchIndex) = patch(:); % unroll in col-major order
        patchIndex = patchIndex + 1;
    end
end
end

