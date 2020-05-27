% Interface for training a GP using the ard Kernel
function [mu,gradient,hessian,cv] = gp_new(X, Y)

[ntrain, d] = size(X);

%% Initial hyperparameter

ell0 = 0.5*sqrt(d);
ell1 = ell0/2;
s0 = std(Y);
sig0 = s0; 
beta = 1e-6;
mean0 = mean(Y);


%% Train GP 
cov = @(hyp) se_kernel(X, hyp);
lmlfun = @(x) lml_exact(cov, Y, x, beta);
hyp = struct('cov', log([ell0, ell1, s0]), 'lik', log(sig0), 'mean', mean0);
params = minimize_quiet(hyp, lmlfun, -120);
sigma = sqrt(exp(2*params.lik) + beta);
fprintf('GP with ard kernel: (ell0, ell1, s, sigma1, mean) = (%.3f, %.3f, %.3f, %.3f, %.3f)\n', exp(params.cov), sigma, params.mean)


%% Calculate interpolation coefficients
sigma2 = sigma^2*ones(1, ntrain);
K = se_kernel(X, params) + diag(sigma2);
lambda = K\(Y-params.mean*ones(ntrain,1));
cv = norm(lambda./diag(inv(K)))^2/ntrain;


%% Function handle returning GP inference to be output
mu = @(XX) inference_mu(XX, X, lambda, params);
gradient = @(XX) inference_gradient(XX, X, lambda, params);
hessian = @(XX) inference_hessian(XX, X, lambda, params);

end



%% prediction
function mu = inference_mu(XX, X, lambda, params)
[m, ~] = size(XX);
K = se_kernel(X, params, XX);
mu = params.mean*ones(m,1) + K*lambda;

end


%% first order derivative
function gradient = inference_gradient(XX, X, lambda, params)
[m, d] = size(XX);
[n, ~] = size(X);
dK = se_kernel_gradient(X, params, XX);

gradient = ones(m,d);
for i=1:d
    gradient(1:m,i) = dK(1:m, 1+(i-1)*n:i*n)*lambda;
end

end


%% second order derivative
function hessian = inference_hessian(XX, X, lambda, params)
[m, d] = size(XX);
[n, ~] = size(X);
d2K = se_kernel_hessian(X, params, XX);

hessian = ones(m*d,d);
for i=1:d
    for j=1:d
        hessian(1+(i-1)*m:i*m,j) = d2K(1+(i-1)*m:i*m, 1+(j-1)*n:j*n)*lambda;            
    end
end
end