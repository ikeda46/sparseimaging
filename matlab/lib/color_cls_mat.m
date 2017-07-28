function [Cls_mat] = color_cls_mat(Cls_mat,Ncl,r,c,Nx,Ny)
% function [Cls_mat] = color_cls_mat(Cls_mat,Ncl,r,c,Nx,Ny)

Cls_mat(r,c) = Ncl;
            
if (r < Nx) && (c < Ny)
    Cls_mat(r+1,c) = Ncl;
    Cls_mat(r,c+1) = Ncl;    
elseif (r == Nx) && (c < Ny)
    Cls_mat(r,c+1) = Ncl;            
elseif (r < Nx) && (c == Ny)               
    Cls_mat(r+1,c) = Ncl;            
end

end