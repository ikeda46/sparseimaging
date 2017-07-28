function [TVij] = TVij_iso(x,Nx,Ny)
% [TVij] = TVij_iso(x,Nx,Ny)
%   Computing isotropic total variation.

tmp = reshape(x,Nx,Ny);

tmp1 = (tmp(1:Nx-1,:)-tmp(2:Nx,:));
tmp2 = (tmp(:,1:Ny-1)-tmp(:,2:Ny));

TVij = zeros(Nx,Ny);

TVij(1:Nx-1,1:Ny-1) = sqrt(tmp1(:,1:Ny-1).^2 + tmp2(1:Nx-1,:).^2);
TVij(1:Nx-1,1:Ny-1) = sqrt(tmp1(:,1:Ny-1).^2 + tmp2(1:Nx-1,:).^2);

TVij(1:Nx-1,Ny) = abs(tmp1(:,Ny));
TVij(Nx,1:Ny-1) = abs(tmp2(Nx,:));

end

