function f = costF_fft_TSV(yax,x,Nx,Ny,lambda2)
% function f = costF_fft_TSV(yAx,x,Nx,Ny,lambda2);

f = norm(yax,'fro')^2/4+lambda2*TSV(x,Nx,Ny);

end