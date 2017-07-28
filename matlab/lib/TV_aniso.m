function [TVcost] = TV_aniso(x,Nx,Ny)
% [TVcost] = TV_aniso(x,Nx,Ny);
%   Computing anisotropic total variation.

tmp = reshape(x,Nx,Ny);

tmp1 = abs(tmp(1:Nx-1,:)-tmp(2:Nx,:));
tmp2 = abs(tmp(:,1:Ny-1)-tmp(:,2:Ny));

TVcost = sum(sum(tmp1)) + sum(sum(tmp2));

end

