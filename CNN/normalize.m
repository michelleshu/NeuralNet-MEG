function data = normalize(data)

% Normalize data to range 0.1 to 0.9 %
dataMin = min(min(min(data)));
data = data - dataMin;

dataMax = max(max(max(data)));
data = data ./ dataMax .* 0.8;
data = data + 0.1;
