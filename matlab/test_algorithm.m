clear

setup

N1 = 128;
N2 = 128;

load testdata128dft
load testdata128fft

lambda1 = [1,0.1,0.01,0.01];
lambda2 = [0.01,0.001,0.0001];

num1 = length(lambda1);
num2 = length(lambda2);

looe = zeros(num1,num2);

tmp = zeros(N1*N2,num1,num2);

xinit = zeros(N1*N2,1);

tmpc = ones(N1,N2);

% tmpc(:,1:40)    = zeros(N1,40);
% tmpc(:,N2-9:N2) = zeros(N1,10);
% tmpc(1:10,:)    = zeros(10,N2);
% tmpc(N1-9:N1,:) = zeros(10,N2);

c_box = reshape(tmpc,N1*N2,1);

for i = 1:num1
    for j = 1:num2      
         tic;
         [x,cost,looe(i,j)] = MFISTA_L1_TSV_nonneg_eht(    y,A,noise_std_dft,N1,N2,xinit,lambda1(i),lambda2(j),5e10);
         toc
         pause
        tic
        [x,cost,LOOE] = MFISTA_L1_TSV_nonneg_eht_fft(y_r,y_i,u_idx,v_idx,noise_std_fft,c_box,N1,N2,xinit,lambda1(i),lambda2(j),5e10);
        toc
        pause
        tmp(:,i,j) = x;
        xinit = x;
    end
end
