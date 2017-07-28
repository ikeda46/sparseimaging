function fftmat = mk_fftmat(u_idx,v_idx,y_r,y_i,N1,N2)
% function fftmat = mk_fftmat(u_idx,v_idx,y,N1,N2)

fftmat = zeros(N1,N2);
M = length(u_idx);

for i = 1:M
    fftmat(u_idx(i),v_idx(i))=y_r(i) + sqrt(-1)*y_i(i);
end

end