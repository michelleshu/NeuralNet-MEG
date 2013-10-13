function printAccuracies(filename)

load(filename);
subjects = {'B', 'D', 'F', 'I', 'J'};
one_means = mean(one_mean_acc, 2);
one_stds = std(one_mean_acc, 0, 2);
two_means = mean(two_mean_acc, 2);
two_stds = std(two_mean_acc, 0, 2);

for i = 1 : numel(subjects)
    fprintf('Subject %s:\n', subjects{i});
    fprintf('1 vs 2: %2.3f(%2.2f)\n', one_means(i), one_stds(i));
    fprintf('2 vs 2: %2.3f(%2.2f)\n', two_means(i), two_stds(i));
end
