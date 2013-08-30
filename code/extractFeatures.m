function features = extractFeatures(img, filters, params)

% 1. convolve input features with kernels
cim = convolve(img,filters,params);

% 2. rectify with absolute val
rim = abs(cim);

% 3. average down sampling
features = avdown(rim, params);

function out = avdown(in, params)
% Separate out sections of timeseries. Then average each section's data by
% brain region

brNames = fieldnames(params.brainRegions); % brain region names
out = zeros(size(in, 1), numel(brNames) * params.numTimeSections);

in_start = 1;    % Where this time section starts in input matrix
out_start = 1;   % Where this time section starts in output matrix
for section = 1 : params.numTimeSections
    for br = 1 : numel(brNames)
        % Collect data from all sensors in this brain region and average
        sensors = params.brainRegions.(brNames{br});
        sensorsData = zeros(size(in, 1), numel(sensors));
        for i = 1 : numel(sensors)
            sensorsData(:, i) = in(:, sensors(i) + in_start - 1);
        end
        out(:, br + out_start - 1) = mean(sensorsData, 2);
    end
    in_start = in_start + params.numSensors;
    out_start = out_start + params.numRegions;
end
 
function out = convolve(in, filters, params)
    % Separate out sections of the time series and vertcat them
    in_rearranged = zeros(params.numSensors * params.numTimeSections, ...
                    params.numSectionTimePoints);

    curr_row = 1; % current row # in rearranged in matrix
    for section = 1 : params.numTimeSections
        in_rearranged(curr_row : curr_row + params.numSensors - 1, :) = ...
            squeeze(in(:, (section - 1) * params.numSectionTimePoints + 1 : ...
            section * params.numSectionTimePoints));
        curr_row = curr_row + params.numSensors;
    end
    out = filters*in_rearranged';