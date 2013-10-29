subjects = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'J'};
rawDataDir = '~/Documents/Mitchell/NeuralNet/data/raw';
pcaDataDir = '~/Documents/Mitchell/NeuralNet/data/pca';

for s = 1 : length(subjects)
    subject = subjects{s};
    
    struct = load(sprintf('%s/%s_raw_avrg.mat', rawDataDir, subject));
    data = struct.data;
    
    %% Isolate time range of interest: stimulus onset (0 ms) to 750 ms
    data = data(:, :, 53 : 202);
    
    %% Generate matrix of examples for PCA
    % We want to find principal components along the sensor dimensions
    % (i.e. want to represent "snapshot" of activations at all sensors at
    %  single time point in <306 dimensions). Therefore, each example
    %  is vector of length 306 (306 is # of sensors).
    
    words = size(data, 1);
    sensors = size(data, 2);
    times = size(data, 3);
   
    % Rearrange data in data matrix X by concatenating all words
    X = zeros(words * times, sensors);
    
    for i = 1 : words
        X((i - 1) * times + 1 : i * times, :) = squeeze(data(i, :, :))';
    end
    
    %% Compute principal components of data matrix
    [coeff, score, eig] = princomp(X);
    
    % Save coeff matrix and eigenvalues
    save(sprintf('%s/%s_coeff.mat', pcaDataDir, subject), 'coeff');
    save(sprintf('%s/%s_eig.mat', pcaDataDir, subject), 'eig');
    
    % Break up data scores by word
    data_pca = zeros(size(data));
    for i = 1 : words
        data_pca(i, :, :) = score((i - 1) * times + 1 : i * times, :)';
    end
    
    % Save PCA scores matrix
    save(sprintf('%s/%s_data_pca.mat', pcaDataDir, subject), 'data_pca');
end
    