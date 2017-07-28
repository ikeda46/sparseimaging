function ddx = d2_sqTV(Nx,Ny)
% function ddx = d2_sqTV(Nx,Ny);

tmp = 8*ones(Nx,Ny);

tmp(1,1)   = 4;
tmp(Nx,1)  = 4;
tmp(1,Ny)  = 4;
tmp(Nx,Ny) = 4;

tmp(2:Nx-1, 1) = tmp(2:Nx-1,1)  - 2;
tmp(2:Nx-1,Ny) = tmp(2:Nx-1,Ny) - 2;
tmp(1, 2:Ny-1) = tmp(1, 2:Ny-1) - 2;
tmp(Nx,2:Ny-1) = tmp(Nx,2:Ny-1) - 2;

ddx = reshape(tmp,Nx*Ny,1);

end