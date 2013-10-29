function inputs = getInputsFromPCA(subject, numComponents, numTimePoints)
% Get compressed PCA representation of data in input format
% numComponents is the number of PCA components to use (reduce sensor dims)
% numTimePoints is the number of averaged time points (reduce time dims)

% Returns inputs, a (w x k)-dimensional data matrix, where w is the number
% of words and k is the total dimensions (time and sensors) used to
% represent each word.

    pcaDataDir = '~/Documents/Mitchell/NeuralNet/data/pca';
    struct = load(sprintf('%s/%s_data_pca.mat', pcaDataDir, subject));
    data = struct.data_pca;
    
    inputs = zeros(size(data, 1), numComponents * numTimePoints);
    
    % Reduce to top PCA components
    data = data(:, 1:numComponents, :);
    
    % Average out time series and transfer results to inputs
    t_size = size(data, 3) / numTimePoints; % size of time block
    for w = 1 : size(data, 1)
        word = squeeze(data(w, :, :));
        for i = 1 : numTimePoints
            slice = word(:, (i - 1) * t_size + 1 : i * t_size);
            slice = mean(slice, 2);
            inputs(w, (i - 1) * numComponents + 1 : i * numComponents) ...
                = slice;
        end
    end

end

