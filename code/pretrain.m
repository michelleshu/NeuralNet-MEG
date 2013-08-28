function [ filters, params ] = pretrain( params )
% Adapted from Richard Socher for use with MEG data

patches = getAllPatches(params);    % Use all patches for pre-training

%% get whitening info from patches
% normalize for contrast
patches = bsxfun(@rdivide, bsxfun(@minus, patches, mean(patches,2)), sqrt(var(patches,[],2)+10));

% Leave out whitening
% C = cov(patches);
% M = mean(patches);
% [V,D] = eig(C);
% P = V * diag(sqrt(1./(diag(D) + 0.1))) * V';

% Now whiten patches before pretraining
% patches = bsxfun(@minus, patches, M) * P;

filters = run_kmeans(patches,params.numFilters);

% params.whiten.P = P;
% params.whiten.M = M;
end

function patches = getAllPatches(params)
% Use ALL patches for pre-training, not random selection like in getPatches

% Allocate space for final patches matrix
patches = zeros(params.numWords * params.numSensors, params.numTimePoints);
for word = 1 : params.numWords
    patches((word - 1) * params.numSensors + 1 : word * params.numSensors, :) = ...
        squeeze(params.data(word, :, :));
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