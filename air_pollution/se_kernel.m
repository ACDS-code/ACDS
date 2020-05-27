function [K, dKhyp] = se_kernel(X, hyp, XX)

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
    dist(:,:,i) = diff_square(:,:,i)/(2*ell(i)^2); %% notice
end
D = sum(dist,3);
K = s^2*exp(-D);


%% Kernel derivative w.r.t params
if nargout == 2
    dKhyp = {1/ell(1)^2*(diff_square(:,:,1).* K), 1/ell(2)^2*(diff_square(:,:,2).* K), 2*K};
end



end