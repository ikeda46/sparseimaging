function [dtX] = dt_X(x, t_list)
% function [dtX] = dt_X(x, t_list);

N_t = size(x,2);

inv_tlist = 1./(t_list(2:N_t) - t_list(1:N_t-1));
d_t = ones(size(x,1),1)*reshape(inv_tlist,1,N_t-1);

dtX = zeros(size(x));

tmp = 2*(x(:,1:N_t-1) - x(:,2:N_t)).*d_t;

dtX(:,1:N_t-1) = tmp;
dtX(:,2:N_t)   = dtX(:,2:N_t) - tmp;

end