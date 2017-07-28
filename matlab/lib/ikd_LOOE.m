function [LOOE,active_x,H_m,A_m] = ikd_LOOE(x,y,A,Nx,Ny,lambda_tv,EPS_tv)
% function [LOOE,active_x,H_m,A_m] = ikd_LOOE(x,y,A,Nx,Ny,lambda_tv,EPS_tv)

tmpx = zeros(size(x));

for i = 1:Nx*Ny
    if abs(x(i)) > EPS_tv
        tmpx(i) = x(i);
    end
end

TV = TVij_iso(tmpx,Nx,Ny);

index_x = find(tmpx~=0)';

A_s = A(:,index_x);

RSS=y-A*tmpx;

[Cls_mat,Cls_rep,Ncl] = find_locked(tmpx,TV,EPS_tv,Nx,Ny);

[H_s,~,~] = compute_full_H_TV(tmpx,index_x,TV,EPS_tv,Nx,Ny);

num = length(index_x);
tmp_active = 1:num;

for i=1:Ncl
    tmp = reshape(Cls_mat,Nx*Ny,1);
    for j = 1:length(index_x)
        if (tmp(index_x(j)) == i) && (index_x(j)~= Cls_rep(i))
            tmp_active(j) = 0;
            num = num-1;
        end
    end
end

j = 0;
active_x = zeros(1,num);
for i = 1:length(tmp_active)
    if tmp_active(i)>0
        j = j + 1;
        active_x(j) = tmp_active(i);
    end
end

for i=1:Ncl
    for j = 1:length(index_x)
        if index_x(j) == Cls_rep(i);
            tmp_rep = j;
            break
        end
    end
        
    tmp = reshape(Cls_mat,Nx*Ny,1);
    for j = 1:length(index_x)
        if (tmp(index_x(j)) == i) && (index_x(j)~= Cls_rep(i))
            %fprintf('%d %d\n',i,index_x(j));
            H_s(tmp_rep,:) = H_s(tmp_rep,:) + H_s(j,:);
            H_s(:,tmp_rep) = H_s(:,tmp_rep) + H_s(:,j);
            
            A_s(:,tmp_rep) = A_s(:,tmp_rep) + A_s(:,j);
        end
    end
end

H_m = H_s(active_x,active_x);
A_m = A_s(:,active_x);

AA_m = A_m'*A_m;
    
Chi = AA_m+ lambda_tv*H_m;

tic
tmp = Chi\A_m';
LOOEfactor = (1-diag(A_m*tmp)).^(-2);
LOOE = mean(LOOEfactor.*RSS.^2)/2;
time1 = toc;
fprintf('LOOE = %g, time = %g\n',LOOE,time1);

end