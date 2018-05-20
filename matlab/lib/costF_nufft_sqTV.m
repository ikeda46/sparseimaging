function f = costF_nufft_sqTV(yax,x,Nx,Ny,lambda2)
% function f = costF_nufft_sqTV(yAx,x,Nx,Ny,lambda,lambda2);

f = real(yax'*yax)/2+lambda2*sqTV(x,Nx,Ny);
%fprintf('costs: %g %g\n',real(yax'*yax)/2,lambda2*sqTV(x,Nx,Ny));

end