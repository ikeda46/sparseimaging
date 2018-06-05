function [x,cost,error,outy] = MFISTA_L1_TSV_eht_nufft(vis,u,v,noise_std,...
                                                      c_box,Nx,Ny,xinit,...
                                                      lambda,lambda2,nonneg_flag,cinit,...
                                                      MAXITER, EPS)
% function [x,cost,error,outy] = MFISTA_L1_TSV_eht_nufft(vis,u,v,th_noise,c_box,Nx,Ny,xinit,lambda,lambda2,cinit,nonneg_flag,MAXITER,EPS);
%
%    y: observed vector
%    u:
%    v:
%    noise_std:
%    c_box:
%    Nx: row number of the image
%    Ny: column number of the image
%    xinit: initial vector for x
%    lambda: lambda for L1
%    lambda2: lambda for sqTV cost
%    nonneg_flag: if nonneg_flag = 1, x is restricted to be nonnegative.
%    c: initial value for the estimate of Lipshitz constant of A'*A
% 
%   This algorithm solves 
%
%    min_x (1/2)||y-A*x||_2^2 + lambda*sum(abs(x)) + lambda2*TSV(x)
%    
%    x is an image with Nx*Ny. x is nonnegative.

%% main loop

MINITER = 100;
tmpcost = zeros(1,MAXITER);
eta = 1.1;
TD = 50;
nufft_eps = 1.0e-12;
nufft_sign = 1;
opts.modeord = 0;

mu = 1;
x = xinit.*c_box;
z = x;

c = cinit;

std_inv = 1./noise_std;

M = length(u);

y_fft = vis.*std_inv;

ynufft = finufft2d2(u,v,nufft_sign,nufft_eps,reshape(x,Nx,Ny),opts);

tmpc = lambda*sum(abs(x))+...
    costF_nufft_TSV((y_fft-ynufft.*std_inv),x,Nx,Ny,lambda2);

for t = 1:MAXITER

    tmpcost(t) = tmpc; 
    
    ynufft = finufft2d2(u,v,nufft_sign,nufft_eps,reshape(z,Nx,Ny),opts);

    yAz = y_fft-ynufft.*std_inv;
    
    if mod(t,10) == 1
        fprintf('%4d cost = %e %e\n',t,tmpcost(t),c);
    end
    
    xnufft = finufft2d1(u,v,(yAz.*std_inv),-1*nufft_sign,nufft_eps,Nx,Ny, ...
                        opts);

    AyAz = reshape(real(xnufft),Nx*Ny,1);
    dvec = d_TSV(z,Nx,Ny);

    AyAz_dvec = AyAz-lambda2*dvec;
    
    Qcore = costF_nufft_TSV(yAz,z,Nx,Ny,lambda2);
 
    for i = 1:MAXITER
        if nonneg_flag == 1
            xtmp = softth_nonneg((AyAz_dvec)/c+z,lambda/c).*c_box;
        else
            xtmp = softth((AyAz_dvec)/c+z,lambda/c).*c_box;
        end
        ynufft = finufft2d2(u,v,nufft_sign,nufft_eps,...
                            reshape(xtmp,Nx,Ny),opts);

        yax  = y_fft-ynufft.*std_inv;
        
        tmpF = costF_nufft_TSV(yax,xtmp,Nx,Ny,lambda2);
        tmpQ = Qcore-(xtmp-z)'*AyAz_dvec+(xtmp-z)'*(xtmp-z)*c/2;
        
        if (tmpF <= tmpQ) 
            break
        end
        c = c*eta;
    end
    
    c = c/eta;
    
    munew = (1+sqrt(1+4*mu^2))/2;
    
    tmpF = tmpF + lambda*sum(abs(xtmp));
    
    if tmpF < tmpcost(t)
        tmpc = tmpF;
        z = (1+(mu-1)/munew)*xtmp + ((1-mu)/munew)*x;
        x = xtmp;
    else
        z = (mu/munew)*xtmp + (1-(mu/munew))*x;
    end
    
    if t>MINITER && (tmpcost(t-TD)-tmpcost(t))<EPS
        break
    end
    
    %% stopping rule
    mu = munew;
    
end

fprintf('converged after %d iterations. cost = %e\n',t,tmpcost(t));

cost = tmpcost(1:t);

show_vlbi_image(x,Nx,Ny)
axis equal tight
drawnow

%% output

outy = finufft2d2(u,v,nufft_sign,nufft_eps,...
                    reshape(x,Nx,Ny),opts);

yax  = y_fft-outy.*std_inv;

error.chisq = real(yax'*yax);
error.l1    = sum(abs(x));
error.TSV   = TSV(x,Nx,Ny);

end