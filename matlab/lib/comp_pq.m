function [p,q] = comp_pq(x,Nx,Ny)
% function [p,q] = comp_pq(x,Nx,Ny)

[r,s] = Lx(x,Nx,Ny);

norm2 = sqrt(r(:,1:Ny-1).^2+s(1:Nx-1,:).^2);

tmp = norm2<1.0e-15;

mask = norm2.*(1-tmp)+tmp;

p = zeros(Nx-1,Ny);
q = zeros(Nx,Ny-1);

p(:,1:Ny-1) = r(:,1:Ny-1)./mask;
q(1:Nx-1,:) = s(1:Nx-1,:)./mask;

p(:,Ny) = sign(r(:,Ny));
q(Nx,:) = sign(s(Nx,:));

end