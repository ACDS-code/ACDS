% Standard Squared Exponential (SE) kernel with gradients
% Input
%     X: training points
%     hyp: hyperparameters
%     XX: testing points (optional)
% Output
%     K: dense kernel matrix

function [K] = se_kernel_2(X, hyp, XX)

ell = exp(hyp.cov(1)); 
s = exp(hyp.cov(2));

[n, d] = size(X);
[n2, ~] = size(XX);

%% (1) Kernel block

D = pdist2(XX, X);
KK(1:n2, 1:n) = exp(-D.^2/(2*ell^2));


%% (4) Hessian block
for i=1:d
    for j=1:d
        K(1+(i-1)*n2:i*n2, 1+(j-1)*n:j*n) = ...
            (-(i==j)/ell^2 + (XX(:,i)-X(:,i)').*(XX(:,j)-X(:,j)')/ell^4).*KK(1:n2, 1:n);
    end
end

K = s^2*K;


end