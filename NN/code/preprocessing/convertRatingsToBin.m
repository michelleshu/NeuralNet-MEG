% Convert semantic feature vector ratings (originally -1 to 1) to binary (0, 1)
% -1, -0.5 become 0
% 0.5, 1 become 1
% If for a particular feature, the mean rating across all words is < 0,
% then 0 becomes 1. Otherwise, stays 0.

sem_matrix_bin = zeros(size(sem_matrix));

% Get the mean ratings for each semantic feature
means = mean(sem_matrix);

for feature = 1 : size(sem_matrix, 2)
    for word = 1 : size(sem_matrix, 1)
        if (sem_matrix(word, feature) < 0)
            sem_matrix_bin(word, feature) = 0;
        elseif (sem_matrix(word, feature) > 0)
            sem_matrix_bin(word, feature) = 1;
        else
            if means(feature) < 0
                sem_matrix_bin(word, feature) = 1;
            end
        end
    end
end
