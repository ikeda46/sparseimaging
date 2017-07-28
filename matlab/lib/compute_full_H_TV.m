function [H_s,Num_x,inv_x] = compute_full_H_TV(x,index_x,TV,EPS_tv,Nx,Ny)
% function [H_s,Num_x,inv_x] = compute_full_H_TV(x,index_x,TV,EPS_tv,Nx,Ny)


Num_x = length(index_x);

inv_x = zeros(1,Nx*Ny);

for i = 1:Num_x
    inv_x(index_x(i))= i;
end

H_s = zeros(Num_x);

[p,q] = comp_pq(x,Nx,Ny);

for i = 1:Nx*Ny
    [tmpx,tmpy] = i2rc(i,Nx);
    if (tmpx < Nx) && (tmpy < Ny)
        if TV(tmpx,tmpy)>EPS_tv
            tmpTV = TV(tmpx,tmpy);
            pij = p(tmpx,tmpy);
            qij = q(tmpx,tmpy);
            
            Cindx = inv_x(rc2i(tmpx,tmpy,Nx,Ny));
            Dindx = inv_x(rc2i(tmpx+1,tmpy,Nx,Ny));
            Rindx = inv_x(rc2i(tmpx,tmpy+1,Nx,Ny));
           
            C = 0;
            D = 0;
            R = 0;
            
            if Cindx > 0
                C = 1;
            end

            if Dindx > 0
                D = 1;
            end
            
            if Rindx > 0
                R = 1;
            end
            
            if C == 1
                tmp = (2-(pij+qij)^2)/tmpTV;
                H_s(Cindx,Cindx) = H_s(Cindx,Cindx) + tmp;
            end
            
            if C == 1 && D ==1
                tmp = (pij*(pij+qij)-1)/tmpTV;
                H_s(Cindx,Dindx) = H_s(Cindx,Dindx) + tmp;
                H_s(Dindx,Cindx) = H_s(Dindx,Cindx) + tmp;
            end
            
            if C == 1 && R ==1
                tmp = (qij*(pij+qij)-1)/tmpTV;
                H_s(Cindx,Rindx) = H_s(Cindx,Rindx) + tmp;
                H_s(Rindx,Cindx) = H_s(Rindx,Cindx) + tmp;
            end
            
            if D == 1
                tmp = (1-pij^2)/tmpTV;
                H_s(Dindx,Dindx) = H_s(Dindx,Dindx) + tmp;
            end

            if R == 1
                tmp = (1-qij^2)/tmpTV;
                H_s(Rindx,Rindx) = H_s(Rindx,Rindx) + tmp;
            end
            
            if D == 1 && R == 1
                tmp = (-pij*qij)/tmpTV;
                H_s(Dindx,Rindx) = H_s(Dindx,Rindx) + tmp;
                H_s(Rindx,Dindx) = H_s(Rindx,Dindx) + tmp;
            end
            
        end
    end
end

end