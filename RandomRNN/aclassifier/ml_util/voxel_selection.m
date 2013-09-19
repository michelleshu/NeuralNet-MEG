function [voxelScores] = voxel_selection(data)
% this will choose the most stable channels, based on Tom's stability
% metric defined in his Science paper. based on Mark's code
% data needs to be voxel X words X repetitions.

numVoxels = size(data,1);

voxelScores = zeros(numVoxels,1);
for v = 1:numVoxels
    if rem(v,10000) == 1,
        fprintf('%s\tvoxel %i\n',datestr(now),v);
    end
    % voxelStabilityMatrix needs to be numPresentations by numWords
    voxelStabilityMatrix = squeeze(data(v,:,:))';
    
    % now we want to compute correlations for each pair of rows in the
    % stability matrix. 'corr' will do this for us on the columns, so
    % we just transpose.
    corrMatrix = corr(voxelStabilityMatrix');
    
    % now we take the mean over
    voxelScores(v) = mean(corrMatrix(find(triu(corrMatrix,1) ~= 0)));
end
