function [ filters, params ] = pretrain( params )
    patches = getFullTimeSeries(params);    % Retrieve all timeseries from data
    patches = bsxfun(@rdivide, bsxfun(@minus, patches, mean(patches,2)), ...
        sqrt(var(patches,[],2)));           % Normalize for contrast
    patches = sectionPatches(patches, params);      % Section off the timeseries

    filters = run_kmeans(patches,params.numFilters);
end


function patches = getFullTimeSeries(params)
% Use ALL patches for pre-training

    % Allocate space for final patches matrix
    patches = zeros(params.numWords * params.numSensors, params.numTotalTimePoints);
    for word = 1 : params.numWords
        patches((word - 1) * params.numSensors + 1 : word * params.numSensors, :) = ...
            squeeze(params.data(word, :, :));
    end
end

function out = sectionPatches(in, params)
% Section off full timeseries into sections and vertically concatenate them

    % Allocate space for final patches matrix
    out = zeros(params.numWords * params.numSensors * params.numTimeSections, ...
        params.numTotalTimePoints / params.numTimeSections);
    
    curr_row = 1;   % keep track of row to write to
    for section = 1 : params.numTimeSections
        out(curr_row : curr_row + (params.numWords * params.numSensors) - 1, :) = ...
            in(:, (section - 1) * params.numSectionTimePoints + 1 : ...
            section * params.numSectionTimePoints);
        curr_row = curr_row + (params.numWords * params.numSensors);
    end
end


% function patches = getPatches(params)
% % Each patch is a single MEG timeseries from one sensor, one word
% % Returns patches of size numPatches x patchLength
% %   (each patch is ultimately a row vector, but added as column vector
% %    and transposed at the end)
% 
% patches = zeros(params.numTimePoints, params.numPTPatches);
% 
% numWant = params.numPTPatches;  % Number of patches we need
% numHave = 0;                    % Number of patches we have collected
% count = 1;
% 
% randSensors = randperm(params.numSensors);  % Sensors in random order
% 
% while numHave < numWant
%     subjData = params.data;
%         
%     % Go to next random sensor in list
%     sensor = randSensors(count);
%     
%     % Collect all word recordings from this sensor (60)
%     for i = 1 : params.numWords
%         patch = squeeze(subjData(i, sensor, :)); % time: 0 to 750 ms
%         patches(:, numHave + 1) = patch;
%         numHave = numHave + 1;
%     end
%     
%     % Go to the next sensor
%     count = count + 1;
% end
% 
% patches = patches';
% end

function fileBool = isValid(name)
fileBool = (~strcmp(name,'.') && ~strcmp(name,'..') && ~strcmp(name,'.DS_Store'));
end