new_data = zeros(60, 306*150);
for word = 1 : 60
    d = data(word, :, :);
    d = d(:);
    new_data(word, :) = d;
end

[weightMatrix, r] = learn_text_from_fmri_kernel_sep_lambda_no_bias(new_data, sem_mat_tool, 1);
   
% For each output time slice, use weight matrix weights to weight the
% inputs of corresponding input nodes

weightedAverages = zeros(306, 150);
weightReshape = reshape(weightMatrix(1:end-1), 306, 150);

for time = 1 : 150
    weights = weightReshape(:, time);
    weightedAv = zeros(306, 1);
    for hn = 1 : 306
        weightedAv = weightedAv + weights(hn) * optInputs(:, hn);
    end
    weightedAv = weightedAv ./ 306;
    weightedAverages(:, time) = weightedAv;
end
