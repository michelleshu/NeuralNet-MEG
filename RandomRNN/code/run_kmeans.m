function centroids = run_kmeans(X, k)

x2 = sum(X.^2,2);
centroids = randn(k,size(X,2))*0.1;%X(randsample(size(X,1), k), :);

% Initialize centroids at random sample of patches
% centroids = X(randsample(size(X, 1), k), :);

BATCH_SIZE=1000;

done = false;
itr = 1;

% for itr = 1:iterations
while ~done
    
    old_centroids = centroids;
    
    c2 = 0.5*sum(centroids.^2,2);
    
    summation = zeros(k, size(X,2));
    counts = zeros(k, 1);
    
    loss =0;
    
    for i=1:BATCH_SIZE:size(X,1)
        lastIndex=min(i+BATCH_SIZE-1, size(X,1));
        m = lastIndex - i + 1;
        
        [val,labels] = max(bsxfun(@minus,centroids*X(i:lastIndex,:)',c2));
        loss = loss + sum(0.5*x2(i:lastIndex) - val');
        
        S = sparse(1:m,labels,1,m,k,m); % labels as indicator matrix
        summation = summation + S'*X(i:lastIndex,:);
        counts = counts + sum(S,1)';
    end
    
    
    centroids = bsxfun(@rdivide, summation, counts);
    
    % just zap empty centroids so they don't introduce NaNs everywhere.
    badIndex = find(counts == 0);
    centroids(badIndex, :) = 0;
    
    diff = max(max(abs(centroids - old_centroids)));
    if diff == 0
        fprintf('K-means converged after %d iterations\n', itr);
        done = true;
    end
    
    itr = itr + 1;
end
