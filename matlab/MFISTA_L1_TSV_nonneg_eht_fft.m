function [x,cost,LOOE] = MFISTA_L1_TSV_nonneg_eht_fft(y_r,y_i,u_idx,v_idx,noise_std,c_box,Nx,Ny,xinit,lambda,lambda2,cinit)
% function [x,cost,LOOE] = MFISTA_L1_TSV_nonneg_eht_fft(y,u_idx,v_idx,th_noise,c_box,Nx,Ny,xinit,lambda,lambda2,cinit);
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
%    c: initial value for the estimate of Lipshitz constant of A'*A
% 
%   This algorithm solves 
%
%    min_x (1/2)||y-A*x||_2^2 + lambda*sum(x) + lambda2*sqTV(x)
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

tmpc = lambda*sum(x)+...
    costF_fft_sqTV((y_fft-(mask.*fft2(reshape(x,Nx,Ny),Nx,Ny))),x,Nx,Ny,lambda2);

for t = 1:MAXITER

    tmpcost(t) = tmpc; 
    
    yAz = y_fft-(mask.*fft2(reshape(z,Nx,Ny),Nx,Ny));
    
    if mod(t,100) == 1
        fprintf('%d cost = %f\n',t,tmpcost(t));
    end
    
    AyAz = reshape(real(ifft2((mask.*reshape(yAz,Nx,Ny)),Nx,Ny)),Nx*Ny,1)/2;
    
    dvec = d_sqTV(z,Nx,Ny);
    
    AyAz_dvec = AyAz-lambda2*dvec;
    
    Qcore = costF_fft_sqTV(yAz,z,Nx,Ny,lambda2);
 
    for i = 1:MAXITER     
        xtmp = softth_nonneg((AyAz_dvec)/c+z,lambda/c).*c_box;

        yax  = y_fft-(mask.*fft2(reshape(xtmp,Nx,Ny),Nx,Ny));
        
        tmpF = costF_fft_sqTV(yax,xtmp,Nx,Ny,lambda2);
        tmpQ = Qcore-(xtmp-z)'*AyAz_dvec+(xtmp-z)'*(xtmp-z)*c/2;
        
        if (tmpF <= tmpQ) 
            break
        end
        c = c*eta;
    end
    
 %   fprintf('hi2\n');
    
    c = c/eta;
    
    munew = (1+sqrt(1+4*mu^2))/2;
    
    tmpF = tmpF + lambda*sum(xtmp);
    
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

% fprintf('Computing approximate LOOE.\n');
% 
% RSS = y-A*x;
% A_s = A(:,x>0);
% 
% tmpdiag = d2_sqTV(Nx,Ny);
% G_s = diag(tmpdiag(x>0));
% 
% Chi = A_s'*A_s + lambda2*G_s;
% 
% LOOE = compute_LOOE(RSS,Chi,A_s);

LOOE = 0;

end