function [x,cost,LOOE] = MFISTA_TV(y,A,Nx,Ny,xinit,lambda_tv,iso_type,cinit)
% function [x,cost,LOOE] = MFISTA_TV(y,A,Nx,Ny,xint,lambda_tv,flag,c);
%
%    y: observed vector
%    A: matrix
%    Nx: row number of the image
%    Ny: column number of the image
%    xinit: initial vector for x
%    lambda_tv: lambda for TV cost
%    iso_type: 'iso' for isotropic TV, 'aniso' for anisotropic TV    
%    c: initial value for the estimate of Lipshitz constant of A'*A
% 
%   This algorithm solves 
%
%    min_x (1/2)||y-A*x||_2^2 + lambda_tv*TV(x)
%    
%    x is an image with Nx*Ny. x can take positive and negative.
%

if strcmp(iso_type, 'aniso')
    flag_iso = 0;
elseif strcmp(iso_type, 'iso')
    flag_iso = 1;
else
    fprintf('iso_type must be ''iso'' or ''aniso.''\n');
    return
end
    

%% main loop

MAXITER = 10000;
MINITER = 100;
tmpcost = zeros(1,MAXITER);
eta = 1.1;
EPS = 1.0e-5;
ITER = 100;
TD = 50;

mu = 1;
x = xinit;
z = x;

if Nx*Ny ~= length(x)
    fprintf('Nx = %d, Ny = %d, but Nx*Ny is not %d.\n',...
           Nx,Ny,length(x));
    return
end

c = cinit;

if flag_iso == 0
    TVcost = TV_aniso(x,Nx,Ny);
else
    TVcost = TV_iso(x,Nx,Ny);
end

tmpc = ((y-A*x)'*(y-A*x)/2)+lambda_tv*TVcost;

for t = 1:MAXITER

    tmpcost(t) = tmpc;
    fprintf('%d cost = %f\n',t,tmpcost(t));

    yAz = y-A*z;
    AyAz = A'*yAz;
    
    Qcore = yAz'*yAz/2;
    
    for i = 1:MAXITER
        
        xtmp = FGP(AyAz/c+z,lambda_tv/c,Nx,Ny,ITER,flag_iso);
        yAxtmp = y-A*xtmp;

        tmpF = yAxtmp'*yAxtmp/2;
        tmpQ = Qcore - (xtmp-z)'*AyAz + (xtmp-z)'*(xtmp-z)*c/2; 

        %        fprintf('Fval = %g, Qval = %g\n',tmpF, tmpQ);
        
        if (tmpF <= tmpQ) 
            break;
        end
        c = c*eta;
    end
    
    c = c/eta;
    
    munew = (1+sqrt(1+4*mu^2))/2;
    
    if flag_iso == 0
        TVcost = TV_aniso(xtmp,Nx,Ny);
    else
        TVcost = TV_iso(xtmp,Nx,Ny);
    end
    
%    fprintf('Total variation is %g\n',TVcost);
    
    if tmpF+lambda_tv*TVcost < tmpcost(t)
        tmpc = tmpF+lambda_tv*TVcost;
        z = (1+(mu-1)/munew)*xtmp + ((1-mu)/munew)*x;
        x = xtmp;
    else
        z = (mu/munew)*xtmp + (1-(mu/munew))*x;
    end
    
    %% stopping rule
    if t>MINITER && (tmpcost(t-TD)-tmpcost(t))<EPS
        break
    end
    
    mu = munew;
end

fprintf('converged after %d iterations. cost = %f\n',t,tmpcost(t));

cost = tmpcost(1:t);

show_vlbi_image(x,Nx,Ny);

%% computing LOOE


EPS_tv = max(x)*1.0e-10;
fprintf('EPS_tv = %g\n',EPS_tv);

[LOOE,~,~,~] = ikd_LOOE(x,y,A,Nx,Ny,lambda_tv,EPS_tv);


end