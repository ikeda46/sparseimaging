function [r,c] = i2rc(i,Nx)
% function [x,y] = i2rc(i,Nx)
    c = ceil(i/Nx);
    r = i-Nx*(c-1);
end

