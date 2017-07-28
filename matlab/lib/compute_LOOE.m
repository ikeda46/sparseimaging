function LOOE = compute_LOOE(RSS,Chi,A)
% function LOOE = compute_LOOE(RSS,Chi,A)

% tic
tmp = Chi\A';
LOOEfactor = (1-diag(A*tmp)).^(-2);
LOOE = mean(LOOEfactor.*RSS.^2)/2;
% time1 = toc;

%fprintf('LOOE = %g, time = %g\n',LOOE,time1);
fprintf('LOOE = %g\n',LOOE);

end