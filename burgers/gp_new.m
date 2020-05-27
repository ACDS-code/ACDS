% Interface for training a GP using the SE Kernel
% 
% [mu, K] = gp_grad(X, Y, DY)
% 
% Input
% X: n by d matrix representing n training points in d dimensions
% Y: training values corresponding Y = f(X)
% Output
% mu: mean function handle such that calling mu(XX) for some predictive points XX calculates the mean of the GP at XX 
% K: dense kernel matrix

function [mu,mu_1,mu_2,sigma3] = gp_new(X, Y)
[ntrain, d] = size(X);

% Initial hyperparameters
ell0 = 0.5*sqrt(d);
s0 = std(Y);
sig0 = 5e-2*s0; 
beta = 1e-6;

% Train GP 
cov = @(hyp) se_kernel(X, hyp);
lmlfun = @(x) lml_exact(cov,Y, x, beta);
hyp = struct('cov', log([ell0, s0]), 'lik', log([sig0]));
params = minimize_quiet(hyp, lmlfun, -50);
sigma = sqrt(exp(2*params.lik) + beta);
fprintf('SE with gradients: (ell, s, sigma1) = (%.3f, %.3f, %.3f)\n', exp(params.cov), sigma)

% Calculate interpolation coefficients
sigma2 = sigma^2*ones(1, ntrain);
K = se_kernel(X, params) + diag(sigma2);
lambda = K\Y;

sigma3 = norm(lambda./diag(inv(K)))^2/ntrain;

% Function handle returning GP mean to be output
mu = @(XX) mean(XX, X, lambda, params);
mu_1 = @(XX) mean_1(XX, X, lambda, params);
mu_2 = @(XX) mean_2(XX, X, lambda, params);

end




function ypred = mean(XX, X, lambda, params)
KK = se_kernel(X, params, XX);
ypred = KK*lambda;

end

function ypred = mean_1(XX, X, lambda, params)
[n, d] = size(X);
[n2, ~] = size(XX);
ypred = ones(n2,d);
KK = se_kernel_1(X, params, XX);
for i=1:d
    ypred(1:n2,i) = KK(1:n2, 1+(i-1)*n:i*n)*lambda;
end
end


function ypred = mean_2(XX, X, lambda, params)
[n, d] = size(X);
[n2, ~] = size(XX);
ypred = ones(n2*d,d);
KK = se_kernel_2(X, params, XX);
for i=1:d
    for j=1:d
        ypred(1+(i-1)*n2:i*n2, j)= KK(1+(i-1)*n2:i*n2, 1+(j-1)*n:j*n)*lambda;
            
    end
end
end