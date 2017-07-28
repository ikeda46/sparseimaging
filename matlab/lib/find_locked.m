function [Cls_mat,Cls_rep,Ncl] = find_locked(x,TV,EPS_tv,Nx,Ny)

Cls_mat = zeros(Nx,Ny);

Ncl = 0;
MASK = zeros(Nx,Ny);
Total = 0;
for r = 1:Nx,
    for c = 1:Ny
        i = rc2i(r,c,Nx,Ny);
        if (TV(r,c)<EPS_tv) && (abs(x(i))>0)
            MASK(r,c) = 1;
            Total = Total + 1;
        end
    end
end

Cls_tmp = zeros(1,Nx*Ny);

while Total > 0
%    fprintf('Total number of not classified small TV pixels are %d\n',Total);

    %% mark first pixel

    for i = 1:(Nx*Ny)
        [r,c] = i2rc(i,Nx);
        if MASK(r,c) == 1
            Ncl = Ncl + 1;
            pivot = i;
            Cls_tmp(Ncl) = pivot;
            
            Cls_mat = color_cls_mat(Cls_mat,Ncl,r,c,Nx,Ny);

            MASK(r,c) = 0;
            Total = Total - 1;
            break
        end
    end

    New_num = 1;

    while (New_num > 0)
        New_num = 0;
        for i = (pivot+1):(Nx*Ny)
            [r,c] = i2rc(i,Nx);
            if (MASK(r,c) == 1) && (r > 1) 
                if ((Cls_mat(r-1,c) == Ncl) && (TV(r-1,c) < EPS_tv))

                    Cls_mat = color_cls_mat(Cls_mat,Ncl,r,c,Nx,Ny);
                    MASK(r,c) = 0;
                    New_num = New_num + 1;
                end
            end
            if (MASK(r,c) == 1) && (c > 1) 
                if ((Cls_mat(r,c-1)) == Ncl && (TV(r,c-1) < EPS_tv))

                    Cls_mat = color_cls_mat(Cls_mat,Ncl,r,c,Nx,Ny);
                    MASK(r,c) = 0;
                    New_num = New_num + 1;
                end
            end
            if (MASK(r,c) == 1) && (r < Nx) 
                if (Cls_mat(r+1,c) == Ncl)

                    Cls_mat = color_cls_mat(Cls_mat,Ncl,r,c,Nx,Ny);                   
                    MASK(r,c) = 0;
                    New_num = New_num + 1;
                end
            end
            if (MASK(r,c) == 1) && (c < Ny) 
                if (Cls_mat(r,c+1) == Ncl)

                    Cls_mat = color_cls_mat(Cls_mat,Ncl,r,c,Nx,Ny);
                    MASK(r,c) = 0;
                    New_num = New_num + 1;
                end
            end
        end
%        fprintf('Newly colored num = %d\n',New_num);
        Total = Total - New_num;
    end
end

Cls_rep = Cls_tmp(1:Ncl);
end