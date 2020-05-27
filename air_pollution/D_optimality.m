function [chosen_index, index_plot]=D_optimality(Theta_true,Theta,chosen_index,n_s,sigma,tol,X)

[m_x,m_y] = size(Theta);
index_plot = [];
alpha = sigma/(sigma+tol);
beta = tol/(sigma+tol);

for i=1:n_s
    Theta_chosen = Theta_true(chosen_index,:);  
    des_mat = Theta_chosen'*Theta_chosen + 0.0000001*eye(m_y);   
    space_filling = zeros(m_x,1);
    D_optimal = zeros(m_x,1);
    D_test = zeros(m_x,1);
    for j=1:m_x
        if ~ismember(j,chosen_index)
            D_optimal(j,1) = 1+Theta(j,:)*(des_mat\Theta(j,:)');
        end
    end
    temp = log(D_optimal);
    % temp = (D_optimal.^beta).*(space_filling.^alpha);
    optimal = max(temp);
    index = find(temp==optimal,1,'first');
    % index = find(temp==optimal);
    chosen_index = [chosen_index;index];
    index_plot = [index_plot;index];
 
end

end


       
        

