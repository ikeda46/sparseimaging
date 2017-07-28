function [sqTVcost] = sqTV(x,Nx,Ny)
% [sqTVcost] = sqTV(x,Nx,Ny);
%   Computing squared total variation.

tmp = reshape(x,Nx,Ny);

tmp1 = (tmp(1:Nx-1,:)-tmp(2:Nx,:));
tmp2 = (tmp(:,1:Ny-1)-tmp(:,2:Ny));

sqTVcost = sum(sum(tmp1.^2)) + sum(sum(tmp2.^2));

end

