function targets = getTargetsUA(subject, featureNums, semMatrixBinFile)
% Retrieve target values for a semantic feature from the sem_matrix
    load(semMatrixBinFile);
    % Load word labels file
    load(sprintf('~/Documents/Mitchell/NeuralNet-MEG/data/unaveraged/labels/%s_labels.mat', ...
        subject));
    
    targets = zeros(numel(labels), numel(featureNums));
    
    for l = 1 : numel(labels)
        % The word label tells us which row of sem_matrix to use
        semRow = labels(l);
    
        for i = 1 : numel(featureNums)
            targets(l, i) = sem_matrix_bin(semRow, featureNums(i));
        end
    end
end