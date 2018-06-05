function f = costF_nufft_TSV(yax,x,Nx,Ny,lambda2)
% function f = costF_nufft_TSV(yAx,x,Nx,Ny,lambda2);

f = real(yax'*yax)/2+lambda2*TSV(x,Nx,Ny);

end