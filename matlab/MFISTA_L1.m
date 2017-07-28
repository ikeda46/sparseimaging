function [x,cost,LOOE] = MFISTA_L1(y,A,xinit,lambda,cinit)
% function [x,cost,LOOE] = MFISTA_L1(y,A,xint,lambda,c);
%
%    y: observed vector
%    A: matrix
%    xinit: initial vector for x
%    lambda: lambda for L1
%    c: initial value for the estimate of Lipshitz constant of A'*A
% 
%   This algorithm solves 
%
%    min_x (1/2)||y-A*x||_2^2 + lambda*sum(abs(x))
%    

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

yAz = y-A*x;

tmpc = yAz'*yAz/2 + lambda*sum(abs(x));

for t = 1:MAXITER

    tmpcost(t) = tmpc;
    fprintf('%d cost = %f\n',t,tmpcost(t));

    yAz = y-A*z;
    AyAz = A'*yAz;
    Qcore = yAz'*yAz/2;
    
    for i = 1:MAXITER     
        xtmp = softth(AyAz/c+z,lambda/c);
        yax = y-A*xtmp;
        tmpF = yax'*yax/2;
        tmpQ = Qcore-(xtmp-z)'*AyAz+(xtmp-z)'*(xtmp-z)*c/2;
 %       fprintf('c = %g, F = %g, Q = %g\n',c,tmpF,tmpQ);
        if (tmpF <= tmpQ) 
            break
        end
        c = c*eta;
    end
    
    c = c/eta;
    
    munew = (1+sqrt(1+4*mu^2))/2;
    
    tmpF = tmpF+lambda*sum(abs(xtmp));
    
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

show_vlbi_image(x,Nsq,Nsq);

%% compute CV

fprintf('Computing approximate LOOE.\n');

RSS = y-A*x;
A_s = A(:,x>0);
Chi = A_s'*A_s;

LOOE = compute_LOOE(RSS,Chi,A_s);

end