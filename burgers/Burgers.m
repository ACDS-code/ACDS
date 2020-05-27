clear all; close all 
clc
%% load data 
load('burgers.mat', 'usol', 't', 'x')
u_star = real(usol)';
t_star = t'; 
x_star = x';   
nsteps = size(t_star,1)-1;
eps = 0.5;

%% Prepare Data
%rng('default')
%i = randi(nsteps);
i = 100;
dt = t_star(i+1) - t_star(i);
du1 = (u_star(:,i+1)-u_star(:,i))/dt;
du1 = du1(2:end-1);
du_true = du1(2:end-1);
x_step = x_star(2)-x_star(1);
ns_max = 70;

%% prepare derivative obervations
y_true = u_star(:,i);
dy_true = (y_true(3:end)-y_true(1:end-2))./(x_star(3:end)-x_star(1:end-2));
d2y_true = (y_true(3:end)-2*y_true(2:end-1)+y_true(1:end-2))/(x_step*x_step);
x_star = x_star(2:end-1);
N_star = size(x_star,1);
y_true = y_true(2:end-1);
Theta_true = pool_data(x_star,y_true,dy_true,d2y_true);
[m_x,m_y] = size(Theta_true);


%% true value
Xi_true = zeros(20,1);
Xi_true(8,1) = -1;
Xi_true(11,1) = .1;
Xi_log = zeros(1,21);
Xi_log(1,8) = 1;
Xi_log(1,11) = 1;


time = 100;
error = zeros(time,2);
error_l0 = zeros(time,2);
sample = zeros(time,2);

for times=1:time

    chosen_col_last = ones(21,1);
    du = du1 + eps*randn(size(du1));
    %% first bunch of data generated at random
    N0 = 5;
    n_s = 20; 
   % chosen_index = randsample(floor(N_star/2), N0);
    chosen_index = randsample(N_star, N0);
    x0 = x_star(chosen_index,:);
    u0 = y_true(chosen_index);
    u0mean = mean(u0);
    u0 = u0-u0mean;
    s = 0;
    while(1)
        
        s = s+1;
        %% GP
        [mu,mu_1,mu_2,sigma] = gp_new(x0,u0);
        sigma = sigma/(std(u0)^2);
        y = mu(x_star)+u0mean;
        dy = mu_1(x_star);
        d2y = mu_2(x_star);
        
        %% pool Data
        [Theta] = pool_data(x_star,y,dy,d2y);
        
        %% sparse regression
        Theta_chosen = Theta_true(chosen_index,:);
        eta = du(chosen_index,:);

        mdl = stepwiselm(Theta_chosen,eta,'Criterion','bic');
        chosen_col = mdl.Formula.InModel;
        Theta1 = Theta_chosen(:,chosen_col);
        mdl = fitlm(Theta1,eta);
        tol = mdl.RMSE;
        tol = tol/std(eta);
        cof = table2array(mdl.Coefficients(2:end,1));

        chosen_col_1 = double(chosen_col)';
        error_1 = calError(chosen_col_1,chosen_col_last);

        if error_1 == 0
            error_2 = norm(cof - cof_last,2)/norm(cof_last,2);
            if error_2 < .0005
                break
            end
        end
        
        chosen_col_last = chosen_col_1;
        cof_last = cof;

        
        %% max sample size
        if s > ns_max
            break
        end
        
        

        %% optimal design
        [chosen_index]=optimal_design(Theta_true,Theta,chosen_index,n_s,sigma,tol,x_star);
        x0 = x_star(chosen_index,:);
        u0 = y_true(chosen_index);
        u0mean = mean(u0);
        u0 = u0-u0mean;
        
        
        
     end   

    Xi = zeros(m_y,1);
    Xi(chosen_col) = cof;
    error(times,1) = calError(chosen_col,Xi_log);
    error_l0(times,1) = norm(Xi-Xi_true,2);
    sample(times,1) = size(chosen_index,1);
  

    
    
    
    sigma = 0;
    tol = 1;
    chosen_col_last = ones(21,1);
    %% first bunch of data generated at random
    chosen_index_1 = randsample(N_star, N0);
    x0 = x_star(chosen_index_1,:);
    u0 = y_true(chosen_index_1);
    u0mean = mean(u0);
    u0 = u0-u0mean;
    s = 0;
    while(1)
    s = s+1;
    %% GP
    [mu,mu_1,mu_2] = gp_new(x0,u0);
    y = mu(x_star)+u0mean;
    dy = mu_1(x_star);
    d2y = mu_2(x_star);

    %% pool data
    Theta = pool_data(x_star,y,dy,d2y);  
    
     %% sparse regression
    Theta_chosen = Theta_true(chosen_index_1,:);
    eta = du(chosen_index_1,:);

    mdl = stepwiselm(Theta_chosen,eta,'Criterion','bic');
    chosen_col = mdl.Formula.InModel;
    Theta1 = Theta_chosen(:,chosen_col);
    mdl = fitlm(Theta1,eta);
    cof = table2array(mdl.Coefficients(2:end,1));

    chosen_col_1 = double(chosen_col)';
    error_1 = calError(chosen_col_1,chosen_col_last);
    
    chosen_col_last = chosen_col_1;
    cof_last = cof;

    if error_1 == 0
        error_2 = norm(cof - cof_last,2)/norm(cof_last,2);
        if error_2 < .0005
            break
        end
    end
    
   %% max sample size
    if s > ns_max
        break
    end
    
    
    
    
    %% optimal design
    [chosen_index_1]=optimal_design(Theta_true,Theta,chosen_index_1,n_s,sigma,tol,x_star);
    x0 = x_star(chosen_index_1,:);
    u0 = y_true(chosen_index_1);
    u0mean = mean(u0);
    u0 = u0-u0mean;
    
   
      
   

    end
    Xi = zeros(m_y,1);
    Xi(chosen_col) = cof;
    error(times,2) = calError(chosen_col,Xi_log);
    error_l0(times,2) = norm(Xi-Xi_true,2);
    sample(times,2) = size(chosen_index_1,1);
    
end


dlmwrite('burgers5.txt',error);
dlmwrite('burgers5.txt',error_l0,'-append');
dlmwrite('burgers5.txt',sample,'-append');

dlmwrite('data5.txt',mean(error));
dlmwrite('data5.txt',std(error),'-append');
dlmwrite('data5.txt',mean(error_l0),'-append');
dlmwrite('data5.txt',std(error_l0),'-append');
dlmwrite('data5.txt',mean(sample),'-append');
dlmwrite('data5.txt',std(sample),'-append');
