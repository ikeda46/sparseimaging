function [TVcost] = TV_iso(x,Nx,Ny)
% [TVcost] = TV(x,Nx,Ny);
%   Computing isotropic total variation.

tmp = reshape(x,Nx,Ny);

tmp1 = (tmp(1:Nx-1,:)-tmp(2:Nx,:));
tmp2 = (tmp(:,1:Ny-1)-tmp(:,2:Ny));

TVcost = sum(sum(sqrt(tmp1(:,1:Ny-1).^2 + tmp2(1:Nx-1,:).^2)))...
    + sum(abs(tmp1(:,Ny))) + sum(abs(tmp2(Nx,:)));

end

