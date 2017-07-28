function [x] = Lpq(p,q,Nx,Ny)
%function x = Lpq(p,q,Nx,Ny);
%   compute x from p and q

tmp = zeros(Nx,Ny);

tmp(1:Nx-1,:) = p;
tmp(2:Nx,:)   = tmp(2:Nx,:) - p;

tmp(:,1:Ny-1) = tmp(:,1:Ny-1) + q;
tmp(:,2:Ny)   = tmp(:,2:Ny) - q;

x = reshape(tmp,Nx*Ny,1);

end

