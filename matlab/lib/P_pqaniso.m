function [r,s] = P_pqaniso(p,q)
%function [r,s] = P_pqaniso(p,q);
%

r = p./max(abs(p),1);
s = q./max(abs(q),1);

end

