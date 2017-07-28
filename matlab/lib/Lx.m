function [p,q] = Lx(x,Nx,Ny)
% [p,q] = Lx(x,N);
%   Compute Lagrange multiplier for TV-lasso.

tmp = reshape(x,Nx,Ny);

p = tmp(1:Nx-1,:)-tmp(2:Nx,:);
q = tmp(:,1:Ny-1)-tmp(:,2:Ny);

end

