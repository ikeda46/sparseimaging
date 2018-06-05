function [x,cost,error,outy] = MFISTA_L1_TSV_eht_fft(y_r,y_i,u_idx,v_idx,noise_std,c_box,Nx,Ny,xinit,lambda,lambda2,nonneg_flag,cinit)
% function [x,cost,error,outy] = MFISTA_L1_TSV_eht_fft(y_r,y_i,u_idx,v_idx,th_noise,c_box,Nx,Ny,xinit,lambda,lambda2,nonneg_flag,cinit);
%
%    y: observed vector
%    u_idx:
%    v_idx:
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

MAXITER = 30000;

MINITER = 100;
tmpcost = zeros(1,MAXITER);
eta = 1.1;
EPS = 1.0e-5;
TD = 50;

sqNN = sqrt(Nx*Ny);

mu = 1;
x = xinit.*c_box;
z = x;

c = cinit;

std_inv = 1./noise_std;

M = length(u_idx);

y_fft = mk_fftmat(u_idx,v_idx,(y_r./noise_std),(y_i./noise_std),Nx,Ny)*sqNN;
mask  = mk_fftmat(u_idx,v_idx,std_inv,zeros(1,M),Nx,Ny);

tmpc = lambda*sum(abs(x))+...
    costF_fft_TSV((y_fft-(mask.*fft2(reshape(x,Nx,Ny),Nx,Ny))),x,Nx,Ny,lambda2);

for t = 1:MAXITER

    tmpcost(t) = tmpc; 
    
    yAz = y_fft-(mask.*fft2(reshape(z,Nx,Ny),Nx,Ny));
    
    if mod(t,100) == 1
        fprintf('%d cost = %g\n',t,tmpcost(t));
    end
    
    AyAz = reshape(real(ifft2((mask.*reshape(yAz,Nx,Ny)),Nx,Ny)),Nx*Ny,1)/2;
    
    dvec = d_TSV(z,Nx,Ny);
    
    AyAz_dvec = AyAz-lambda2*dvec;
    
    Qcore = costF_fft_TSV(yAz,z,Nx,Ny,lambda2);
 
    for i = 1:MAXITER
        if nonneg_flag == 1
            xtmp = softth_nonneg((AyAz_dvec)/c+z,lambda/c).*c_box;
        else
            xtmp = softth((AyAz_dvec)/c+z,lambda/c).*c_box;
        end

        yax  = y_fft-(mask.*fft2(reshape(xtmp,Nx,Ny),Nx,Ny));
        
        tmpF = costF_fft_TSV(yax,xtmp,Nx,Ny,lambda2);
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


fprintf('converged after %d iterations. cost = %f\n',t,tmpcost(t));

cost = tmpcost(1:t);

show_vlbi_image(x,Nx,Ny)
drawnow

%% compute CV

outy = fft2(reshape(x,Nx,Ny),Nx,Ny);

yAz = y_fft-(mask.*outy);

error.chisq = costF_fft_TSV(yAz,z,Nx,Ny,0);
error.l1    = sum(abs(x));
error.TSV   = TSV(x,Nx,Ny);

end