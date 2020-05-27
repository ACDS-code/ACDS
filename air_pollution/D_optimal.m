clear all
close all
clc


%% load data
% load advection-dispersion;
load heat
[xx,yy]=meshgrid(x,y);
index = 50;
eps = 0.2;


%% prepare derivative 
q = 2:99;
xx = xx(q,q);
yy = yy(q,q);

U = T(q,q,index);
dUdt1 = (T(q,q,index+1) - T(q,q,index-1))/(2*dt);
dUdx = (T(q+1,q,index)-T(q-1,q,index))/(2*dx);
dUdy = (T(q,q+1,index)-T(q,q-1,index))/(2*dy);
d2Udx2 = (T(q+1,q,index)-2*T(q,q,index)+T(q-1,q,index))/(dx^2);
d2Udy2 = (T(q,q+1,index)-2*T(q,q,index)+T(q,q-1,index))/(dy^2);
d2Udxdy = (T(q+1,q+1,index) + T(q-1,q-1,index) ...
          - T(q+1,q-1,index) - T(q-1,q+1,index))/(4*dx*dy);

      
%% candidate pool
id = 3:3:96;
xx = xx(id,id); yy = yy(id,id);
xx = xx(:); yy = yy(:); X = [xx yy];

U = U(id,id); dUdt1 = dUdt1(id,id); dUdx = dUdx(id,id); dUdy = dUdy(id,id);
d2Udx2 = d2Udx2(id,id); d2Udy2 = d2Udy2(id,id); d2Udxdy = d2Udxdy(id,id);
U = U(:); dUdt1 = dUdt1(:); dUdx = dUdx(:); dUdy = dUdy(:);
d2Udx2 = d2Udx2(:); d2Udy2 = d2Udy2(:); d2Udxdy = d2Udxdy(:);


%% pool data
[Theta_true] = pool_data(xx,U,dUdx,dUdy,d2Udx2,d2Udy2,d2Udxdy);
N_star = size(U,1);
N0 = 16;    % intial sample size
n_s = 16;   % batch size
ns_max = 6;    % max iteration times

%% prepare true value
[m_x,m_y] = size(Theta_true);   % true value (l2 norm)
Xi_true = zeros(m_y,1);
Xi_true(11,1) = 1;
Xi_true(12,1) = 1;

Xi_log = zeros(m_y+1,1);    % true Boolean value (false positive + false negative)
Xi_log(11,1) = 1;
Xi_log(12,1) = 1;

%% record the comparison criteria
chosen_col_last = ones(m_y+1,2);
time = 50; % repeat times
error = zeros(time,1);
error_l0 = zeros(time,1);
sample = zeros(time,1);

for times =1:time
    dUdt = dUdt1 + eps*randn(size(dUdt1));
    s = 0;
    
    %% intial design
    % N_star_square= reshape(1:N_star,[32,32]);
    % chosen_index = N_star_square(2:7:32,2:7:32);
    ii = [26,6; 10,16; 30,22; 4,30; 8,24; 32,8; 20,13; 14,20; 28,28; 18,26; 11,10; 2,11; 22,4; 5,1; 16,31; 23,18];
    chosen_index = ii(:,1) + (ii(:,2)-1)*32;
    chosen_index = chosen_index(:);   
    
    x0 = X(chosen_index,:);
    u0 = U(chosen_index,:);
      
    
    %% sequential design
    while (1)
        s = s+1;
        
        %% train GP
        [mu,gradient,hessian,cv] = gp_new(x0,u0);  
        
        %% mean
        u1 = mu(X);
        
        %% gradient
        du1 = gradient(X);
        dudy = du1(:,1);
        dudx = du1(:,2);
       
        %% hessian 
        d2u1 = hessian(X);
        d2udy2 = d2u1(1:N_star,1);
        d2udx2 = d2u1(N_star+1:2*N_star,2);
        d2udxdy = d2u1(1:N_star,2);
        
        %% leave one out cross validation for GP
        sigma = cv/(std(u0)^2);
  
        %% pool Data
        [Theta] = pool_data(xx,u1,dudx,dudy,d2udx2,d2udy2,d2udxdy);
        
        %% sparse regression
        Theta_chosen = Theta_true(chosen_index,:);
        eta = dUdt(chosen_index,:);

        mdl = stepwiselm(Theta_chosen,eta,'Criterion','bic');
        chosen_col_1 = mdl.Formula.InModel;
        Theta1 = Theta_chosen(:,chosen_col_1);
        mdl = fitlm(Theta1,eta);
        tol = mdl.RMSE;
        tol = tol/std(eta);
        cof = table2array(mdl.Coefficients(2:end,1));

        chosen_col = double(chosen_col_1)' ;
        error_1 = calError(chosen_col,chosen_col_last);
        
        
        if error_1 == 0
            error_2 = norm(cof - cof_last,2)/norm(cof_last,2);
           
            if error_2 < 1e-9
                break
            end
        end
        
        %% max sample size
        if s > ns_max
            break
        end

         chosen_col_last = chosen_col;
         cof_last = cof;
        
        
        %% optimal design
        fprintf('(error_regression, error_gp) = (%.3f, %.3f)\n', tol, sigma)
        [chosen_index, index_plot]=D_optimality(Theta_true,Theta,chosen_index,n_s,sigma,tol,X);
        x0 = X(chosen_index,:);
        u0 = U(chosen_index,:);
   
    end   

    z1 = zeros(m_y,1);
    z1(chosen_col_1) = cof;
    Xi = z1;
    error(times,1) = calError(chosen_col,Xi_log);
    error_l0(times,1) = norm(Xi-Xi_true,2);
    sample(times,1) = size(chosen_index,1);
  
    
end


dlmwrite('D_optimal2.txt',error);
dlmwrite('D_optimal2.txt',error_l0,'-append');
dlmwrite('D_optimal2.txt',sample,'-append');

dlmwrite('D_optimal2.txt',mean(error));
dlmwrite('D_optimal2.txt',std(error),'-append');
dlmwrite('D_optimal2.txt',mean(error_l0),'-append');
dlmwrite('D_optimal2.txt',std(error_l0),'-append');
dlmwrite('D_optimal2.txt',mean(sample),'-append');
dlmwrite('D_optimal2.txt',std(sample),'-append');


