function [r,s] = P_pqiso(p,q,Nx,Ny)
%function [r,s] = P_pqiso(p,q,Nx,Ny);
%   P_p

tmp = sqrt(p(:,1:Ny-1).^2+q(1:Nx-1,:).^2);

mask(:,1:(Ny-1)) = max(tmp(:,1:(Ny-1)),1);
mask(:,Ny) = max(abs(p(:,Ny)),1);
r = p./mask;

clear mask
mask(1:(Nx-1),:) = max(tmp(1:(Nx-1),:),1);
mask(Nx,:) = max(abs(q(Nx,:)),1);
s = q./mask;

end

