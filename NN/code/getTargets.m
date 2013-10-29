function targets = getTargets(featureNums, semMatrixBinFile)
% Retrieve target values for a semantic feature from the sem_matrix
    load(semMatrixBinFile);
    targets = zeros(size(sem_matrix_bin, 1), numel(featureNums));
    for i = 1 : numel(featureNums)
        targets(:, i) = sem_matrix_bin(:, featureNums(i));
    end
end