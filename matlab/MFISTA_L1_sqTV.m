function [x,cost,LOOE] = MFISTA_L1_sqTV(y,A,Nx,Ny,xinit,lambda,lambda2,cinit)
% function [x,cost,LOOE] = MFISTA_L1_sqTV(y,A,xint,lambda,lambda2,c);
%
%    y: observed vector
%    A: matrix
%    Nx: row number of the image
%    Ny: column number of the image
%    xinit: initial vector for x
%    lambda: lambda for L1
%    lambda2: lambda for sqTV cost
%    c: initial value for the estimate of Lipshitz constant of A'*A
% 
%   This algorithm solves 
%
%    min_x (1/2)||y-A*x||_2^2 + lambda*sum(abs(x)) + lambda2*sqTV(x)
%    
%    x is an image with Nx*Ny.

%% main loop

MAXITER = 10000;
MINITER = 100;
tmpcost = zeros(1,MAXITER);
eta = 1.1;
EPS = 1.0e-5;
TD = 50;

mu = 1;
x = xinit;
z = x;

Nsq = sqrt(length(x));

c = cinit;

tmpc = lambda*sum(abs(x))+costF_sqTV((y-A*x),x,Nx,Ny,lambda2);

for t = 1:MAXITER

    tmpcost(t) = tmpc;
    fprintf('%d cost = %f\n',t,tmpcost(t));

    yAz = y-A*z;
    AyAz = A'*yAz;
    dvec = d_sqTV(z,Nx,Ny);
    
    AyAz_dvec = AyAz-lambda2*dvec;
    
    Qcore = costF_sqTV(yAz,z,Nx,Ny,lambda2);
    
    for i = 1:MAXITER     
        xtmp = softth((AyAz_dvec)/c+z,lambda/c);
        yax = y-A*xtmp;
        tmpF = costF_sqTV(yax,xtmp,Nx,Ny,lambda2);
        tmpQ = Qcore-(xtmp-z)'*AyAz_dvec+(xtmp-z)'*(xtmp-z)*c/2;
        %fprintf('c = %g, F = %g, Q = %g\n',c,tmpF,tmpQ);
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

%% compute CV

fprintf('Computing approximate LOOE.\n');

RSS = y-A*x;
A_s = A(:,abs(x)>0);

tmpdiag = d2_sqTV(Nx,Ny);
G_s = diag(tmpdiag(abs(x)>0));
Chi = A_s'*A_s + lambda2*G_s;

LOOE = compute_LOOE(RSS,Chi,A_s);

end