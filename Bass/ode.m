clear all
close all
clc
addpath('./utils');

%% prepare 
trial = 50;
eps = 0.04; 
polyorder = 5;  % search space up to fifth order polynomials
usesine = 0;    % no trig functions

tspan=.01:.01:30;   % time span
pool_size = size(tspan,2);

ns_max = 100;
k = 16;         % initial sample size
n_s = 16;      % one_shot sample size

%% record data
error_gamma = zeros(trial,1);
error_l2 = zeros(trial,1);
sample = zeros(trial,1);

%% repeat from here 
for rep=1:trial
    
    p = 0.03*rand(1);       
    q = 0.2*rand(1)+0.3;      
    % p = 0.03; 
    % q = 0.4;
       

    %% ODE solution
    xs = (1-exp(-(p+q)*tspan'))./(1+q/p*exp(-(p+q)*tspan'));
    
    %% compute Derivative 
    dxs = -q*xs.^2+(q-p)*xs+p;
    dxs = dxs + eps*randn(size(dxs));  % add noise

    Theta_true = [ones(size(xs)) xs xs.^2 xs.^3 xs.^4 xs.^5];
    [m_x,m_y] = size(Theta_true);

    %% prepare true value
    Xi_true = zeros(m_y,1);
    Xi_true(1,1) = p;   
    Xi_true(2,1) = q-p;  
    Xi_true(3,1) = -q; 
    

    Xi_log = zeros(m_y+1,1);
    Xi_log(1,1) = 1;
    Xi_log(2,1) = 1;
    Xi_log(3,1) = 1;
    

    %% initial design  
    chosen_index = randi(pool_size,k,1);
    time = chosen_index/100;
    x = xs(chosen_index,:);
    
    s = 0;
    Xi_last = zeros(6,1);
   
    %% sequential experiment 
    while(1)
        
        s = s+1;
        %% Gaussian process
        [mu,~,~,sigma] = gp_new(time,x);
        y_predict = mu(tspan');
        sigma = sigma/std(x)^2;
        
        %% pool Data  (i.e., build library of nonlinear time series)
        Theta_predict = [ones(size(y_predict)) y_predict y_predict.^2 y_predict.^3 y_predict.^4 y_predict.^5];
        Theta_now = Theta_true(chosen_index,:);
        dx = dxs(chosen_index,:);

        %% compute Sparse regression
        mdl = stepwiselm(Theta_now,dx,'Criterion','bic','Intercept',false);
        chosen_col = mdl.Formula.InModel;
        Theta = Theta_now(:,chosen_col);
        mdl = fitlm(Theta,dx,'Intercept',false);
        tol = mdl.RMSE;
        tol = tol/std(dx);
        cof_1 = table2array(mdl.Coefficients(:,1));  % the first coefficient is intercerpt
        z = zeros(m_y,1);
        z(chosen_col) = cof_1;
        
        chosen_col = double(chosen_col)';
        Xi = z;
  
               
        %% convergence check
        error = norm(Xi - Xi_last,2)/norm(Xi_last,2);
            if error < 1e-2
                break
            end
            
        %% max sample size
        if s > ns_max
            break
        end

        %% ACDS      
        [chosen_index] = acds(Theta_true,Theta_predict,chosen_index,n_s,sigma,tol,tspan');
        time = chosen_index/100;
        x = xs(chosen_index,:);
        Xi_last = Xi;
       

    end
    
    error_gamma(rep,1) = calError(chosen_col,Xi_log);
    error_l2(rep,1) = norm(Xi-Xi_true,2);
    sample(rep,1) = size(chosen_index,1);
   
end



%% output
dlmwrite('data4.txt',mean(error_gamma));
dlmwrite('data4.txt',std(error_gamma),'-append');
dlmwrite('data4.txt',mean(error_l2),'-append');
dlmwrite('data4.txt',std(error_l2),'-append');
dlmwrite('data4.txt',mean(sample),'-append');
dlmwrite('data4.txt',std(sample),'-append');
