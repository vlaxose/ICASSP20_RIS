function [h_conv,h_prop] = estimateChannel(snr_db)

addpath('basic_system_functions');
addpath(genpath('benchmark_algorithms'));

%% Parameter initialization
Nt = 1;
Nr = 32;
Gr = Nr;
Gt = Nt;
total_num_of_clusters = 1;
total_num_of_rays = 3;
L = 1;
Lr = 8;
T = Nr*20;
square_noise_variance = 10^(-snr_db/10);
Imax = 50;
numOfnz = 10;



%% System model
[H,Zbar,Ar,At,Dr,Dt] = wideband_mmwave_channel(L, Nr, Nt, total_num_of_clusters, total_num_of_rays, Gr, Gt, 'ULA');

% Additive white Gaussian noise
N = sqrt(square_noise_variance/2)*(randn(Nr, T) + 1j*randn(Nr, T));
Psi_i = zeros(T, T, Nt);
% Generate the training symbols
for k=1:Nt
% 4-QAM symbols
s = qam4mod([], 'mod', T);
Psi_i(:,:,k) =  toeplitz(s);
Psi_i(:,:,k) = randn(size(Psi_i(:,:,k))) + 1j*randn(size(Psi_i(:,:,k)));
end
%    
%    
%          
%% Conventional HBF 
T_conv = round(T/Nr);
[Y_hbf_nr, W_c, Psi_bar] = hbf(H, N(:, 1:T_conv), Psi_i(1:T_conv,1:T_conv,:), T_conv, Nr, createBeamformer(Nr, 'quantized'));
A = W_c'*Dr;
B = zeros(L*Gt, T_conv);
for l=1:L
  B((l-1)*Gt+1:l*Gt, :) = Dt'*Psi_bar(:,:,l);
end    
%     Phi = kron((B).', A);
y = vec(Y_hbf_nr);

% LS based
S_ls = pinv(A)*Y_hbf_nr*pinv(B);

% OMP with MMV based
s_omp_solver = spx.pursuit.joint.OrthogonalMatchingPursuit(A, numOfnz);
S_omp_mmv = s_omp_solver.solve(Y_hbf_nr*pinv(B));
h_conv = A*S_omp_mmv.Z*B;

%% Proposed HBF
T_prop = round(T/Lr);
W = createBeamformer(Nr, 'quantized');    
[Y_proposed_hbf, W_tilde, Psi_bar, Omega, Y] = proposed_hbf(H, N(:, 1:T_prop), Psi_i(1:T_prop,1:T_prop,:), T_prop, Nr, Lr, W);
tau_Y = 1/norm(Y_proposed_hbf, 'fro')^2;
tau_Z = 1/norm(Zbar, 'fro')^2/2;
eigvalues = eigs(Y_proposed_hbf'*Y_proposed_hbf);
rho = sqrt(min(eigvalues)*(1/norm(Y_proposed_hbf, 'fro')^2));

A = W_tilde'*Dr;
B = zeros(L*Gt, T_prop);
for l=1:L
  B((l-1)*Gt+1:l*Gt, :) = Dt'*Psi_bar(:,:,l);
end    
S_proposed = proposed_algorithm(Y_proposed_hbf, Omega, A, B, Imax, tau_Y, tau_Z, rho, 'approximate');
h_prop = A*S_proposed*B;
  

end

