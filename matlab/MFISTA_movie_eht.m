function [x,cost,terms,outvis] = MFISTA_movie_eht(vis, u, v, noise_std, time_idx,...
                                                  c_box, Nx, Ny, xinit, ...
                                                  lambda, lambda2, lambda_t, nonneg_flag,...
                                                  cinit, MAXITER, EPS)
% function [x,cost,terms,outvis] = MFISTA_movie_eht(vis, u, v, noise_std, time_idx,...
%                                                  c_box, Nx, Ny, xinit, ...
%                                                  lambda, lambda2, lambda_t, nonneg_flag,...
%                                                  cinit, MAXITER, EPS)
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
%    lambda2: lambda for TSV cost
%    lambda_t: lambda for time 
%    c: initial value for the estimate of Lipshitz constant of A'*A
% 
%   This algorithm solves 
%
%    min_x (1/2)||y-A*x||_2^2 + lambda*sum(x) + lambda2*TSV(x)
%    
%    x is an image with Nx*Ny. x is nonnegative.

MINITER = 100;
tmpcost = zeros(1,MAXITER);
eta = 1.1;
TD = 50;
nufft_eps = 1.0e-12;
nufft_sign = 1;
opts.modeord = 0;

c = cinit;

mu = 1;

%% initialize data

M = length(u);
N_t = max(time_idx(:,1));

x = (xinit.*c_box)*ones(1,N_t);
z = x;
AyAz = x;

vnum = zeros(1,N_t);

for i = 1:M
    vnum(time_idx(i,1)) = vnum(time_idx(i,1)) + 1;
end

t_list = zeros(1,N_t);

if size(time_idx,2) == 2
    for i = 1:M
        t_list(time_idx(i,1)) = t_list(time_idx(i,1)) + time_idx(i,2);
    end
    for i = 1:N_t
        t_list(i) = t_list(i)/vnum(i);
    end
    
    tmp = max(t_list)-min(t_list);
    t_list = N_t*t_list/tmp;
else
    t_list = 1:N_t;
end

for i = 1:N_t
    data(i).wvis    = zeros(vnum(i),1);
    data(i).idx     = zeros(vnum(i),1);
    data(i).std_inv = zeros(vnum(i),1);
    data(i).u       = zeros(vnum(i),1);
    data(i).v       = zeros(vnum(i),1);
end

idx = zeros(1,N_t);

for i = 1:M
    t_i = time_idx(i,1);
    idx(t_i) = idx(t_i) +1;
    
    data(t_i).wvis(idx(t_i))    = vis(i)/noise_std(i);
    data(t_i).idx(idx(t_i))     = i;
    data(t_i).std_inv(idx(t_i)) = 1/noise_std(i);
    data(t_i).u(idx(t_i))       = u(i);
    data(t_i).v(idx(t_i))       = v(i);
end

tmpc = 0;

for i = 1:N_t
    ynufft = finufft2d2(data(i).u, data(i).v, nufft_sign, ...
                        nufft_eps, reshape(x(:,i),Nx,Ny), opts);

    tmpc = tmpc + lambda*sum(abs(x(:,i))) + ... 
           costF_nufft_TSV((data(i).wvis - ynufft.*(data(i).std_inv)), ... 
                            x(:,i), Nx, Ny, lambda2);
end

%% main loop

for t = 1:MAXITER

    tmpcost(t) = tmpc; 

    if mod(t,10) == 1
        fprintf('%4d cost = %e %e\n',t,tmpcost(t),c);
    end

    Qcore = 0;
    dTSV = zeros(size(x));

    for i = 1:N_t
        ynufft = finufft2d2(data(i).u, data(i).v, ...
                            nufft_sign, nufft_eps, ...
                            reshape(z(:,i),Nx,Ny), opts); 

        yAz = data(i).wvis - ynufft.* (data(i).std_inv);
        
        Qcore = Qcore + costF_nufft_TSV(yAz,z(:,i),Nx,Ny,lambda2);

        xnufft = finufft2d1(data(i).u, data(i).v, ...
                            (yAz.*(data(i).std_inv)), ...
                            -1*nufft_sign, nufft_eps, Nx, Ny, opts);
       
        AyAz(:,i) = reshape(real(xnufft),Nx*Ny,1);
        
        dTSV(:,i) = d_TSV(z(:,i), Nx, Ny);
    end

    Qcore = Qcore + lambda_t * d_X(z, t_list);

    dtX = dt_X(z, t_list);
    
    AyAz_dvec = AyAz - lambda2*dTSV - lambda_t*dtX;
    
    for ii = 1:MAXITER
        if nonneg_flag == 1
            xtmp = softth_nonneg((AyAz_dvec)/c+z,lambda/c).*(c_box* ...
                                                             ones(1,N_t));
        else
            xtmp = softth((AyAz_dvec)/c+z,lambda/c).*(c_box*ones(1,N_t));
        end

        tmpF = 0;
        tmpQ = Qcore;

        for i =1:N_t
            ynufft = finufft2d2(data(i).u, data(i).v, ...
                                nufft_sign, nufft_eps, ...
                                reshape(xtmp(:,i),Nx,Ny), opts);

            yax  = data(i).wvis - ynufft.* (data(i).std_inv);
        
            tmpF = tmpF + costF_nufft_TSV(yax,xtmp(:,i), Nx, Ny, lambda2);
            tmpQ = tmpQ - (xtmp(:,i)-z(:,i))'*AyAz_dvec(:,i) + ...
                   (xtmp(:,i)-z(:,i))'*(xtmp(:,i)-z(:,i))*c/2;
        end
        
        tmpF = tmpF + lambda_t * d_X(xtmp,t_list);

        if (tmpF <= tmpQ) 
            break
        end
        c = c*eta;
        if c> 1.0e200
            break
        end
    end
    
    c = c/eta;
    
    munew = (1+sqrt(1+4*mu^2))/2;
    
    tmpF = tmpF + lambda*sum(sum(abs(xtmp)));
    
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

for i = 1:N_t
    show_vlbi_image(x(:,i),Nx,Ny);
    axis equal tight
    drawnow()
    %    pause
end

%% compute model.vis

terms = zeros(1,4);

error.chisq = 0;
error.l1    = 0;
error.TSV   = 0;

outvis = zeros(M,1);

for i = 1:N_t
    ynufft = finufft2d2(data(i).u, data(i).v, ...
                        nufft_sign, nufft_eps, ...
                        reshape(x(:,i),Nx,Ny), opts);

    yax  = data(i).wvis - ynufft.* (data(i).std_inv);
    
    for j = 1:idx(i)
        outvis(data(i).idx(j)) = ynufft(j);
    end

    error.chisq = error.chisq + real(yax'*yax);
    error.l1    = error.l1 + sum(abs(x(:,i)));
    error.TSV   = error.TSV + TSV(x(:,i), Nx,Ny);

end

error.dX   = d_X(x, t_list);

end