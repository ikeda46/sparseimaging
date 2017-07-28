function [x] = FGP_L1(b,lambda_1,lambda_tv,Nx,Ny,ITER,flag_iso)
% function [x] = FGP_L1(b,lambda_1,lambda_tv,Nx,Ny,ITER.flag_iso);
%    approximate optimization

p = zeros(Nx-1,Ny);
q = zeros(Nx,Ny-1);
r = p;
s = q;

t = 1;
for iter = 1:ITER
    
    tnew = (1+sqrt(1+4*t^2))/2;
        
    tmp = softth(b-lambda_tv*Lpq(r,s,Nx,Ny),lambda_1);
    
    [tmpr,tmps] = Lx(tmp,Nx,Ny);
    
    if flag_iso == 0
        [pnew,qnew] = P_pqaniso((r+tmpr/(8*lambda_tv)),...
            (s+tmps/(8*lambda_tv)));
    else
        [pnew,qnew] = P_pqiso((r+tmpr/(8*lambda_tv)),...
            (s+tmps/(8*lambda_tv)),Nx,Ny);
    end
    
    r = pnew*((t-1+tnew)/tnew) - p*((t-1)/tnew);
    s = qnew*((t-1+tnew)/tnew) - q*((t-1)/tnew);
    
    t = tnew;   
    p = pnew;
    q = qnew;
end

x = softth(b-lambda_tv*Lpq(p,q,Nx,Ny),lambda_1);

end