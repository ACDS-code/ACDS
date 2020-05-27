function [Theta] = pool_data(x,y,dy,d2y)
%% pool data
Theta = [ones(size(x))  y   y.^2  y.^3    dy   dy.*dy  dy.*dy.*dy  y.*dy   y.*y.*dy  y.*dy.*dy...
    d2y  d2y.*d2y   d2y.*d2y.*d2y  y.*d2y   y.*y.*d2y  y.*d2y.*d2y  dy.*d2y   dy.*dy.*d2y  dy.*d2y.*d2y  y.*dy.*d2y];
end
