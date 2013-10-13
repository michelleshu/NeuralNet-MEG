subject = 'J';

load(sprintf('./data/%s_raw_avrg.mat', subject));
data = data(:, :, 53:202);
% Normalize
dataMax = max(max(max(data)));
data = data ./ dataMax .* 0.8;
data = data + 0.1;

load(sprintf('./results/network/%s_1.000e-01R_100K.mat', subject));

error_norms = zeros(size(data, 1), 1);  % norm of error for all words
diff = zeros(size(data, 1), 1); % abs difference between input, output

for word = 1 : 60
    input = squeeze(data(word, :, :));
    hidden = zeros(size(W1, 1), size(data, 3));
    output = zeros(size(input));
    
    for time = 1 : 150
        i = squeeze(input(:, time));
        
        % Forward propagate from input to hidden layer
        hidden(:, time) = sigmoid(W1 * i + b1);

        % Forward propagate from hidden to output layer
        output(:, time) = sigmoid(W2 * hidden(:, time) + b2);
    end
    
    error_norms(word) = norm(input - output);
    diff(word) = mean(mean(abs(input - output)));
end

disp(mean(error_norms));
disp(mean(diff));