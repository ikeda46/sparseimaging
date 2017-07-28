function xnew = softth_nonneg(v,eta)
% function xnew = softth_nonneg(v,eta);

xnew = ((v-eta)>0).*(v-eta);
