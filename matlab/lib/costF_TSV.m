function f = costF_TSV(yax,x,Nx,Ny,lambda2)
% function f = costF_TSV(yAx,x,Nx,Ny,lambda2);

f = yax'*yax/2 +lambda2*TSV(x,Nx,Ny);

end