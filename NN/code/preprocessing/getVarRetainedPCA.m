subjects = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'J'};
totalComponents = 306;
pcaDataDir = '~/Documents/Mitchell/NeuralNet/data/pca';

var_matrix = zeros(length(subjects), totalComponents);

for i = 1 : length(subjects)
    subject = subjects{i};
    eig = load(sprintf('%s/%s_eig.mat', pcaDataDir, subject));
    eig = eig.eig;
    
    for j = 1 : totalComponents
        var_matrix(i, j) = sum(eig(1:j)) / sum(eig);
    end
end

save(sprintf('%s/var_retained.mat', pcaDataDir), 'var_matrix');