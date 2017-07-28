function xnew = softth(v,eta)
% function xnew = softth(v,eta);

xnew = ((abs(v)-eta)>0).*(v-sign(v)*eta);
