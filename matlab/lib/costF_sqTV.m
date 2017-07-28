function f = costF_sqTV(yax,x,Nx,Ny,lambda2)
% function f = costF_sqTV(yAx,x,Nx,Ny,lambda,lambda2);

f = yax'*yax/2 +lambda2*sqTV(x,Nx,Ny);

end