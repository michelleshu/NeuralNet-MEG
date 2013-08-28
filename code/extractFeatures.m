function features = extractFeatures(img, filters, params)

% 1. convolve input features with kernels
cim = convolve(img,filters,params);


% 2. rectify with absolute val
rim = abs(cim);

% Leave out for now
% 3. local normalization
% lnim = localnorm(rim);

% 4. average down sampling
features = avdown(rim, params);

function out = avdown(in, params)
% Pool by brain regions of sensors
brNames = fieldnames(params.brainRegions);

out = zeros(size(in, 1), numel(brNames));
for br = 1 : numel(brNames)
    % Collect data from all sensors in this brain region and average
    sensors = params.brainRegions.(brNames{br});
    sensorsData = zeros(size(in, 1), numel(sensors));
    for i = 1 : numel(sensors)
        sensorsData(:, i) = in(:, sensors(i));
    end
    out(:, br) = mean(sensorsData, 2);
end
    
% function out = avdown(in, params)
% % Pool by just averaging; original uses convolution
% % Input has dimensions K x # sensors
% compressedLength = 10;   % Hard code dimensions to reduce # sensors to
% 
% in = in';
% gs = floor(size(in, 1) / compressedLength); % group size: # rows to average
% 
% out = zeros(compressedLength, size(in, 2));
% for i = 1 : compressedLength
%     out(i, :) = mean(in((i - 1) * gs + 1 : (i * gs), :));
% end
% out = out';


function out = convolve(patches,filters,params)
% Let patches be the MEG image for one word. Use all channels for now.

% whiten patches
% patches = bsxfun(@rdivide, bsxfun(@minus, patches, mean(patches,2)), sqrt(var(patches,[],2)+10));
% patches = bsxfun(@minus, patches, params.whiten.M) * params.whiten.P;

out = filters*patches';

function lnim = localnorm(in)
%
% given a set of feature maps, performs local normalization
% out = (in -mean(in))/std(in)
% mean and std are defined over local neighborhoods that span
% all feature maps and a local spatial neighborhood
%

filtSize = [1 9];
k = fspecial('gaussian',filtSize,1.591);

ker = zeros(size(in,1), size(k,2));
for i = 1:size(in,1)
    ker(i,:) = k(:);
end
ker = ker / sum(ker(:));

% mu = E(in)
inmean = lnormconvolve(in, ker);

% in-mu
inzmean = in - inmean;
% (in - mu)^2
inzmeansq = inzmean .* inzmean;
% std = sqrt ( E (in - mu)^2 )
instd = sqrt(lnormconvolve(inzmeansq,ker));
% threshold std with the mean
mstd = mean(instd(:));
instd(instd<mstd) = mstd;
% scale the input by (in - mu) / std
lnim = inzmean ./ instd;

function out = lnormconvolve(in, ker)

out = zeros(size(in));
for i=1:size(in,1)
    cin = in(i,:);
    cker = ker(i,:);
    out(i,:) = conv(cin, cker, 'same');
end

sout = sum(out);
for i=1:size(out,1)
    out(i,:) = sout(:);
end
