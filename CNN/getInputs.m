function inputs = getInputs(subject, timeSize)
% Get compressed PCA representation of data in input format
% numComponents is the number of PCA components to use (reduce sensor dims)
% numTimePoints is the number of averaged time points (reduce time dims)

% Returns inputs, a (w x k)-dimensional data matrix, where w is the number
% of words and k is the total dimensions (time and sensors) used to
% represent each word.

    rawDataDir = '../data/raw';
    struct = load(sprintf('%s/%s_raw_avrg.mat', rawDataDir, subject));
    data = struct.data;
    
    % Isolate time range of interest: stimulus onset (0 ms) to 750 ms
    data = data(:, :, 53 : 202);
    
    grad1 = data(:, 1:3:306, :);
    grad2 = data(:, 2:3:306, :);
    mag = data(:, 3:3:306, :);

    % Normalize sets of sensors independently
    grad1 = normalize(grad1);
    grad2 = normalize(grad2);
    mag = normalize(mag);

    % Replace data matrix with normalized results
    data(:, 1:102, :) = grad1;
    data(:, 103:204, :) = grad2;
    data(:, 205:306, :) = mag;
    
    inputs = zeros(size(data, 1), size(data, 2), timeSize);
    
    % Average out time series and transfer results to inputs
    t_size = size(data, 3) / timeSize; % size of time block
    for w = 1 : size(data, 1)
        word = squeeze(data(w, :, :));
        for i = 1 : timeSize
            slice = word(:, (i - 1) * t_size + 1 : i * t_size);
            slice = mean(slice, 2);
            inputs(w, :, i) = slice;
        end
    end

end

