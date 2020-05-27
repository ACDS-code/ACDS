function [dK] = se_kernel_gradient(X, hyp, XX)

if nargin == 2
    XX = X;
end

[m, d] = size(XX);
[n, ~] = size(X);
ell = exp(hyp.cov(1:end-1));
s = exp(hyp.cov(end));


diff = zeros(m, n, d);
for i=1:d
    diff(:,:,i) = XX(:,i)-X(:,i)';
end      
diff_square = diff.^2;


dist = zeros(m, n, d);
%% Kernel block
for i=1:d
    dist(:,:,i) = diff_square(:,:,i)/(2*ell(i)^2);
end
D = sum(dist,3);
K = s^2*exp(-D);


%% Gradient block
for i=1:d
    dK(1:m, 1+(i-1)*n:i*n) = -diff(:,:,i)/ell(i)^2.*K;
end


end