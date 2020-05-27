% Standard Squared Exponential (SE) kernel with gradients
% Input
%     X: training points
%     hyp: hyperparameters
%     XX: testing points (optional)
% Output
%     K: dense kernel matrix

function [K] = se_kernel_1(X, hyp, XX)

ell = exp(hyp.cov(1)); 
s = exp(hyp.cov(2));

[n, d] = size(X);
[n2, ~] = size(XX);

%% (1) Kernel block

D = pdist2(XX, X);
KK(1:n2, 1:n) = exp(-D.^2/(2*ell^2));


%% (2) Gradient block
for i=1:d
    K(1:n2, 1+(i-1)*n:i*n) = (XX(:,i) - X(:,i)')/ell^2.*KK(1:n2, 1:n);
end

K = -s^2*K;

end