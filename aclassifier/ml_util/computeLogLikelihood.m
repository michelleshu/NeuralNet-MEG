function [ll] = computeLogLikelihood(x,mu,sigma)

% Compute the individual log-likelihood terms.
z = (x - mu) ./ sigma;
ll = -.5.*z.*z - log(sqrt(2.*pi).*sigma);
