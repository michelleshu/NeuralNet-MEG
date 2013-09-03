K_options = [100, 150, 200];
R_options = [50, 100];

A_acc = zeros(numel(K_options) * 2 + 1, numel(R_options));
%B_acc = zeros(numel(K_options) * 2 + 1, numel(R_options));
C_acc = zeros(numel(K_options) * 2 + 1, numel(R_options));
%D_acc = zeros(numel(K_options) * 2 + 1, numel(R_options));
E_acc = zeros(numel(K_options) * 2 + 1, numel(R_options));
%F_acc = zeros(numel(K_options) * 2 + 1, numel(R_options));
G_acc = zeros(numel(K_options) * 2 + 1, numel(R_options));
%I_acc = zeros(numel(K_options) * 2 + 1, numel(R_options));
%J_acc = zeros(numel(K_options) * 2 + 1, numel(R_options));

for k_index = 1 : numel(K_options)
    K = K_options(k_index);
    
    for r_index = 1 : numel(R_options)
        R = R_options(r_index);
        
        load(sprintf('acc_%dK_%dR.mat', K, R));
        ones = mean(one_mean_acc, 2);
        twos = mean(two_mean_acc, 2);
        
        A_acc(k_index, r_index) = ones(1);
        %B_acc(k_index, r_index) = ones(2);
        C_acc(k_index, r_index) = ones(2);
        %D_acc(k_index, r_index) = ones(4);
        E_acc(k_index, r_index) = ones(3);
        %F_acc(k_index, r_index) = ones(6);
        G_acc(k_index, r_index) = ones(4);
        %I_acc(k_index, r_index) = ones(8);
        %J_acc(k_index, r_index) = ones(9);
        
        k2 = k_index + numel(K_options) + 1;
        A_acc(k2, r_index) = twos(1);
        %B_acc(k2, r_index) = twos(2);
        C_acc(k2, r_index) = twos(2);
        %D_acc(k2, r_index) = twos(4);
        E_acc(k2, r_index) = twos(3);
        %F_acc(k2, r_index) = twos(6);
        G_acc(k2, r_index) = twos(4);
        %I_acc(k2, r_index) = twos(8);
        %J_acc(k2, r_index) = twos(9);
    end
end

csvwrite('A.csv', A_acc);
%csvwrite('B.csv', B_acc);
csvwrite('C.csv', C_acc);
%csvwrite('D.csv', D_acc);
csvwrite('E.csv', E_acc);
%csvwrite('F.csv', F_acc);
csvwrite('G.csv', G_acc);
%csvwrite('I.csv', I_acc);
%csvwrite('J.csv', J_acc);