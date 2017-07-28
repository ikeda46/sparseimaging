function f = costF_fft_sqTV(yax,x,Nx,Ny,lambda2)
% function f = costF_sqTV(yAx,x,Nx,Ny,lambda,lambda2);

f = norm(yax,'fro')^2/4+lambda2*sqTV(x,Nx,Ny);

end