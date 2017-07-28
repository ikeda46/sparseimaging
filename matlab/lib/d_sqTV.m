function [dvec] = d_sqTV(x,Nx,Ny)
% [sqTVcost] = d_sqTV(x,Nx,Ny);
%   Computing squared total variation.

tmp = reshape(x,Nx,Ny);

tmp1 = 2*(tmp(1:Nx-1,:)-tmp(2:Nx,:));
tmp2 = 2*(tmp(:,1:Ny-1)-tmp(:,2:Ny));

dtmp = zeros(Nx,Ny);

dtmp(1:Nx-1,:) = dtmp(1:Nx-1,:) + tmp1;
dtmp(2:Nx,:)   = dtmp(2:Nx,:)   - tmp1;

dtmp(:,1:Ny-1) = dtmp(:,1:Ny-1) + tmp2;
dtmp(:,2:Ny)   = dtmp(:,2:Ny)   - tmp2;

dvec = reshape(dtmp,Nx*Ny,1);

end

