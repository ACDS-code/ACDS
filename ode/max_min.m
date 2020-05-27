function [chosen_index]=max_min(m_x,chosen_index,n_s,X)

for i=1:n_s   
    space_filling = zeros(m_x,1);   
    for j=1:m_x
        if ~ismember(j,chosen_index)
            space = X(j,:)-X(chosen_index,:);
            space_filling(j,1) = min(sqrt(sum(space.^2,2)));
        end
    end  
    optimal = max(space_filling);
    index = find(space_filling==optimal,1,'first');
    chosen_index = [chosen_index;index];
end

end


       
        

