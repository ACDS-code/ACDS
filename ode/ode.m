clear all
close all
clc
addpath('./utils');

%% prepare 
trial = 50;
eps = .2; 
polyorder = 5;  % search space up to fifth order polynomials
usesine = 0;    % no trig functions
n = 2;          % 2D system
tspan=.01:.01:30;   % time span
pool_size = size(tspan,2);
x0 = [2; 0];        % initial conditions
ns_max = 20;
k = 16;         % initial sample size
n_s = 16;      % one_shot sample size

%% record data
error_gamma = zeros(trial,1);
error_l2 = zeros(trial,1);
sample = zeros(trial,1);

%% repeat from here 
for rep=1:trial
    
    % a = -rand(1)-0.5;     
    a = -.5;
    % b = rand(1)+2;        
    b = 2;
    A = [a b;
        -b a];  
    rhs = @(x)A*x;         % ODE right hand side

    %% ODE solver
    options = odeset('RelTol',1e-10,'AbsTol',1e-10*ones(1,n));
    [t,xs]=ode45(@(t,x)rhs(x),tspan,x0,options);  % integrate
    
    %% compute Derivative 
    dxs = xs*A';
    dxs = dxs + eps*randn(size(dxs));  % add noise

    Theta_true = poolData(xs,n,polyorder,usesine);
    [m_x,m_y] = size(Theta_true);

    %% prepare true value
    Xi_true = zeros(m_y,2);
    Xi_true(2,1) = a;   % Xi_true(2,1) = -.5;
    Xi_true(3,1) = b;   % Xi_true(3,1) = 2;
    Xi_true(2,2) = -b;  % Xi_true(2,2) = -2;
    Xi_true(3,2) = a;   % Xi_true(3,2) = -.5;

    Xi_log = zeros(m_y+1,2);
    Xi_log(2,1) = 1;
    Xi_log(3,1) = 1;
    Xi_log(2,2) = 1;
    Xi_log(3,2) = 1;

    %% initial design  
    chosen_index = randi(pool_size,k,1);
    time = chosen_index/100;
    x = xs(chosen_index,:);
    
    s = 0;
    Xi_last = zeros(21,2);
   
    %% sequential experiment 
    while(1)
        
        s = s+1;
        %% Gaussian process
        [mu,~,~,sigma1] = gp_new(time,x(:,1));
        y_predict(:,1) = mu(tspan');
        sigma1 = sigma1/std(x(:,1))^2;
        [mu,~,~,sigma2] = gp_new(time,x(:,2));
        y_predict(:,2) = mu(tspan');
        sigma2 = sigma2/std(x(:,2))^2;
        sigma = (sigma1+sigma2)/2;

        %% pool Data  (i.e., build library of nonlinear time series)
        Theta_predict = poolData(y_predict,n,polyorder,usesine);
        Theta_now = Theta_true(chosen_index,:);
        dx = dxs(chosen_index,:);

        %% compute Sparse regression
        mdl = stepwiselm(Theta_now,dx(:,1),'Criterion','bic');
        chosen_col_1 = mdl.Formula.InModel;
        Theta = Theta_now(:,chosen_col_1);
        mdl = fitlm(Theta,dx(:,1));
        tol_1 = mdl.RMSE;
        tol_1 = tol_1/std(dx(:,1));
        cof_1 = table2array(mdl.Coefficients(2:end,1));  % the first coefficient is intercerpt
        z1 = zeros(m_y,1);
        z1(chosen_col_1) = cof_1;

        mdl = stepwiselm(Theta_now,dx(:,2),'Criterion','bic');
        chosen_col_2 = mdl.Formula.InModel;
        Theta = Theta_now(:,chosen_col_2);
        mdl = fitlm(Theta,dx(:,2));
        tol_2 = mdl.RMSE;
        tol_2 = tol_2/std(dx(:,2));
        cof_2 = table2array(mdl.Coefficients(2:end,1));  % the first coefficient is intercerpt
        z2 = zeros(m_y,1);
        z2(chosen_col_2) = cof_2;

        chosen_col = [double(chosen_col_1)' double(chosen_col_2)'];
        Xi = [z1 z2];
        tol = (tol_1+tol_2)/2;
               
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
dlmwrite('data2.txt',mean(error_gamma));
dlmwrite('data2.txt',std(error_gamma),'-append');
dlmwrite('data2.txt',mean(error_l2),'-append');
dlmwrite('data2.txt',std(error_l2),'-append');
dlmwrite('data2.txt',mean(sample),'-append');
dlmwrite('data2.txt',std(sample),'-append');
