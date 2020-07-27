clear;
clc;

%% Parameter initialization
Nt = 1;
Nr = 32;
Gr = Nr;
Gt = Nt;
total_num_of_clusters = 1;
total_num_of_rays = 3;
Np = total_num_of_clusters*total_num_of_rays;
L = 1;
maxMCRealizations = 10;
Lr = 2;
T_range = Nr*[5 8 11 14 ];
square_noise_variance = 10^(-5/10);
Imax = 50;
numOfnz = 10;

%% Variables initialization
error_proposed = zeros(maxMCRealizations,1);
error_proposed_angles = zeros(maxMCRealizations,1);
error_ls = zeros(maxMCRealizations,1);
error_omp_nr = zeros(maxMCRealizations,1);
error_svt = zeros(maxMCRealizations,1);
error_vamp = zeros(maxMCRealizations,1);
error_cosamp = zeros(maxMCRealizations,1);
error_omp_mmv = zeros(maxMCRealizations,1);
error_tssr = zeros(maxMCRealizations,1);
mean_error_proposed = zeros(length(T_range),1);
mean_error_proposed_angles = zeros(length(T_range),1);
mean_error_ls =  zeros(length(T_range),1);
mean_error_omp_nr =  zeros(length(T_range),1);
mean_error_svt =  zeros(length(T_range),1);
mean_error_vamp =  zeros(length(T_range),1);
mean_error_cosamp =  zeros(length(T_range),1);
mean_error_omp_mmv =  zeros(length(T_range),1);
mean_error_tssr =  zeros(length(T_range),1);

%% Iterations for different SNRs, training length and MC realizations
for t_indx = 1:length(T_range)
  T = T_range(t_indx);

  for r=1:maxMCRealizations
      
   disp(['T  = ', num2str(T), ', realization: ', num2str(r)]);

   
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
    error_ls(r) = norm(S_ls-Zbar)^2/norm(Zbar)^2;
    if(error_ls(r)>1)
        error_ls(r)=1;
    end

    % OMP with MMV based
    s_omp_solver = spx.pursuit.joint.OrthogonalMatchingPursuit(A, numOfnz);
    S_omp_mmv = s_omp_solver.solve(Y_hbf_nr*pinv(B));
    error_omp_mmv(r) = norm(S_omp_mmv.Z-Zbar)^2/norm(Zbar)^2;
    if(error_omp_mmv(r)>1)
        error_omp_mmv(r)=1;
    end
    
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
    [S_proposed, Y_proposed] = proposed_algorithm(Y_proposed_hbf, Omega, A, B, Imax, tau_Y, tau_Z, rho, 'approximate');
    error_proposed(r) = norm(S_proposed-Zbar)^2/norm(Zbar)^2;
    if(error_proposed(r)>1)
        error_proposed(r)=1;
    end

  end
  
    mean_error_proposed(t_indx) = mean(error_proposed);
    mean_error_proposed_angles(t_indx) = mean(error_proposed_angles);
    mean_error_ls(t_indx) = mean(error_ls);
    mean_error_omp_nr(t_indx) = mean(error_omp_nr);
    mean_error_svt(t_indx) = mean(error_svt);   
    mean_error_vamp(t_indx) = mean(error_vamp);
    mean_error_cosamp(t_indx) = mean(error_cosamp);
    mean_error_omp_mmv(t_indx) = mean(error_omp_mmv);
    mean_error_tssr(t_indx) = mean(error_tssr);

end


figure;
p = semilogy(T_range, mean_error_ls);hold on;
set(p, 'LineWidth',2, 'LineStyle', ':', 'Color', 'Black');
% p = semilogy(T_range, mean_error_svt);hold on;
% set(p, 'LineWidth',1, 'LineStyle', '-', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'Black', 'Marker', '>', 'MarkerSize', 6, 'Color', 'Black');
% p = semilogy(T_range, mean_error_omp_nr);hold on;
% set(p,'LineWidth',1, 'LineStyle', '-', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'Black', 'Marker', '<', 'MarkerSize', 6, 'Color', 'Black');
% p = semilogy(T_range, mean_error_vamp);hold on;
% set(p,'LineWidth',1, 'LineStyle', '-', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'Black', 'Marker', 'o', 'MarkerSize', 6, 'Color', 'Black');
% p = semilogy(T_range, mean_error_cosamp);hold on;
% set(p,'LineWidth',1, 'LineStyle', '-', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'White', 'Marker', 'o', 'MarkerSize', 6, 'Color', 'Black');
p = semilogy(T_range, mean_error_omp_mmv);hold on;
set(p,'LineWidth',1, 'LineStyle', '-', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'White', 'Marker', '+', 'MarkerSize', 6, 'Color', 'Black');
% p = semilogy(T_range, mean_error_tssr);hold on;
% set(p, 'LineWidth',1, 'LineStyle', '-', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'Black', 'Marker', 'x', 'MarkerSize', 6, 'Color', 'Black');
p = semilogy(T_range, (mean_error_proposed));hold on;
set(p,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Blue', 'MarkerFaceColor', 'Blue', 'Marker', 'h', 'MarkerSize', 8, 'Color', 'Blue');
% p = semilogy(T_range, (mean_error_proposed_angles));hold on;
% set(p,'LineWidth',2, 'LineStyle', '--', 'MarkerEdgeColor', 'Green', 'MarkerFaceColor', 'Green', 'Marker', 's', 'MarkerSize', 8, 'Color', 'Green');

% legend({'LS', 'SVT', 'OMP', 'VAMP', 'CoSaMP', 'MMV-OMP', 'TSSR',  'Proposed', 'Proposed with angle information'}, 'FontSize', 12, 'Location', 'Best');
legend({'LS','OMP-MMV', 'Proposed'}, 'FontSize', 12, 'Location', 'Best');

xlabel('Number of Training Symbols');
ylabel('NMSE')
grid on;set(gca,'FontSize',11);
savefig(['results/nmse_Nr',num2str(Nr),'_L',num2str(L),'.fig'])
