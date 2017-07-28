function [x,cost,LOOE] = MFISTA_L1_TSV_nonneg_eht(y,A,std_vec,Nx,Ny,xinit,lambda,lambda2,cinit)
% function [x,cost,LOOE] = MFISTA_L1_TSV_nonneg_eht(y,A,std_vec,Nx,Ny,xint,lambda,lambda2,c);
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
%    min_x (1/2)||y-A*x||_2^2 + lambda*sum(x) + lambda2*sqTV(x)
%    
%    x is an image with Nx*Ny. x is nonnegative.

%% main loop

MAXITER = 30000;
% MAXITER = 10;
MINITER = 100;
tmpcost = zeros(1,MAXITER);
eta = 1.1;
EPS = 1.0e-5;
TD = 50;

mu = 1;
x = xinit;
z = x;

newA = diag((1./std_vec))*A;
A = newA;

newY = y./std_vec;
y = newY;

clear newA newY;

c = cinit;

tmpc = lambda*sum(x)+costF_sqTV((y-A*x),x,Nx,Ny,lambda2);

for t = 1:MAXITER

    tmpcost(t) = tmpc;
    if mod(t,100) == 1
        fprintf('%d cost = %f\n',t,tmpcost(t));
  %      fprintf('         %f\n',(y-A*x)'*(y-A*x)/2);
    end
    
    yAz = y-A*z;
    AyAz = A'*yAz;
    
%     if t == 1
%         fprintf('%f\n',sum(abs(AyAz.^2)))
%         figure(1)
%         colormap gray
%         imagesc(reshape(AyAz,Nx,Ny));axis equal off
%     end
%     
%     figure(2)
    
    dvec = d_sqTV(z,Nx,Ny);
    
    AyAz_dvec = AyAz-lambda2*dvec;
    
     if t == 1
         fprintf('AyAz max %f min %f pow %f\n',...
            max(AyAz),min(AyAz),norm(AyAz,2)^2);
         fprintf('dvec max %f min %f pow %f\n',...
            max(dvec),min(dvec),norm(dvec,2)^2);
        fprintf('max %f min %f pow %f\n',...
            max(AyAz_dvec),min(AyAz_dvec),norm(AyAz_dvec,2)^2);
        save('AyAz.mat','AyAz')
     end
    
    Qcore = costF_sqTV(yAz,z,Nx,Ny,lambda2);
    
 %   fprintf('Qcore = %f\n',Qcore);
    
    for i = 1:MAXITER     
        xtmp = softth_nonneg((AyAz_dvec)/c+z,lambda/c);
        yax = y-A*xtmp;
        
 %       fprintf('norm of yax is %f\n',norm(yax,'fro')^2);
        
        tmpF = costF_sqTV(yax,xtmp,Nx,Ny,lambda2);
        tmpQ = Qcore-(xtmp-z)'*AyAz_dvec+(xtmp-z)'*(xtmp-z)*c/2;

%         fprintf('xtmp = %g\n',xtmp'*A'*A*xtmp);
%         fprintf('yax = %f\n',yax'*yax);
%         fprintf('c = %g, F = %g, Q = %g, Q-F = %f\n',...
%                 c,tmpF,tmpQ,tmpQ-tmpF);

        if (tmpF <= tmpQ) 
            break
        end
        c = c*eta;
    end
 
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
LOOE  = 0;

end