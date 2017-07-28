function [xpos] = P_positive(x)
%function xpos = P_positive(x);
%   threshold with 0

xpos = ceil(0.9*sign(x)).*x;

end

