function [ f ] = costL1(yAx,x,lambda)
% function [ f ] = costL1(y,A,x,lambda);
%  compute cost for L1 and TV

f = yAx'*yAx/2+lambda*sum(x);

end

