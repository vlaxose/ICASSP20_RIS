function rates()
% Nof_samples: number of channel samples

clc



Nof_samples = 500;
N = 32;            % Number of RIS elements

snr_db_range = [-15:5:15];
for snr = 1:length(snr_db_range)
  
SNR = 10^(snr_db_range(snr)/10);
% h_1 = sqrt(SNR/2)*(randn(N,1,Nof_samples)+1i*randn(N,1,Nof_samples)); % all available channels between the N-element RIS and the single-antenna BS
for i=1:Nof_samples
  h_1(:,:,i) = sqrt(SNR/2)*wideband_mmwave_channel(1, N, 1, 1, 3, N, 1, 'ULA');
end
% h_2 = sqrt(SNR/2)*(randn(1,N,Nof_samples)+1i*randn(1,N,Nof_samples));  % all available channels between the single-antenna UE and the N-element RIS 
for i=1:Nof_samples
  h_2(:,:,i) = sqrt(SNR/2)*wideband_mmwave_channel(1, 1, N, 1, 3, 1, N, 'ULA');
end

%% Perfect CSI
instantaneous_rates_b_1 = zeros(1,Nof_samples);
instantaneous_rates_b_2 = zeros(1,Nof_samples);
available_phases_b_1 = linspace(0,pi,2); % b=1
available_phases_b_2 = linspace(0,pi,4); % b=2
closest_theta_b_1 = zeros(1,N);
closest_theta_b_2 = zeros(1,N);
for i = 1 : Nof_samples
    for n = 1 : N
        theta_opt = angle(h_1(n,1,i)*h_2(1,n,i));
        [~,index_b_1] = min(abs(available_phases_b_1-theta_opt));
        [~,index_b_2] = min(abs(available_phases_b_2-theta_opt));
        closest_theta_b_1(n) = available_phases_b_1(index_b_1);
        closest_theta_b_2(n) = available_phases_b_2(index_b_2);
    end
    Phi_b_1 = diag(exp(-1i*closest_theta_b_1));
    Phi_b_2 = diag(exp(-1i*closest_theta_b_2));
    instantaneous_rates_b_1(i) = log2(1+abs(h_2(1,:,i)*Phi_b_1*h_1(:,1,i))^2);
    instantaneous_rates_b_2(i) = log2(1+abs(h_2(1,:,i)*Phi_b_2*h_1(:,1,i))^2);
end
Rate_b_1(snr) = mean(instantaneous_rates_b_1);
Rate_b_2(snr) = mean(instantaneous_rates_b_2);

%% Estimated CSI with Conventional Technique
for i=1:Nof_samples
  [H1_conv, ~] = estimateChannel(snr);
  [H2_conv, ~] = estimateChannel(snr);
  
  h_1(:,:,i) = sqrt(SNR/2)*vec(H1_conv(:,1));
  h_2(:,:,i) = sqrt(SNR/2)*vec(H2_conv(:,1));
end
instantaneous_rates_b_1 = zeros(1,Nof_samples);
instantaneous_rates_b_2 = zeros(1,Nof_samples);
available_phases_b_1 = linspace(0,pi,2); % b=1
available_phases_b_2 = linspace(0,pi,4); % b=2
closest_theta_b_1 = zeros(1,N);
closest_theta_b_2 = zeros(1,N);
for i = 1 : Nof_samples
    for n = 1 : N
        theta_opt = angle(h_1(n,1,i)*h_2(1,n,i));
        [~,index_b_1] = min(abs(available_phases_b_1-theta_opt));
        [~,index_b_2] = min(abs(available_phases_b_2-theta_opt));
        closest_theta_b_1(n) = available_phases_b_1(index_b_1);
        closest_theta_b_2(n) = available_phases_b_2(index_b_2);
    end
    Phi_b_1 = diag(exp(-1i*closest_theta_b_1));
    Phi_b_2 = diag(exp(-1i*closest_theta_b_2));
    instantaneous_rates_b_1(i) = log2(1+abs(h_2(1,:,i)*Phi_b_1*h_1(:,1,i))^2);
    instantaneous_rates_b_2(i) = log2(1+abs(h_2(1,:,i)*Phi_b_2*h_1(:,1,i))^2);
end
Rate_b_1_est_conv(snr) = mean(instantaneous_rates_b_1);
Rate_b_2_est_conv(snr) = mean(instantaneous_rates_b_2);

%% Estimated CSI with Proposed Technique
for i=1:Nof_samples
  [~, H1_prop] = estimateChannel(snr);
  [~, H2_prop] = estimateChannel(snr);
  
  h_1(:,:,i) = sqrt(SNR/2)*vec(H1_prop(:,1));
  h_2(:,:,i) = sqrt(SNR/2)*vec(H2_prop(:,1));
end
instantaneous_rates_b_1 = zeros(1,Nof_samples);
instantaneous_rates_b_2 = zeros(1,Nof_samples);
available_phases_b_1 = linspace(0,pi,2); % b=1
available_phases_b_2 = linspace(0,pi,4); % b=2
closest_theta_b_1 = zeros(1,N);
closest_theta_b_2 = zeros(1,N);
for i = 1 : Nof_samples
    for n = 1 : N
        theta_opt = angle(h_1(n,1,i)*h_2(1,n,i));
        [~,index_b_1] = min(abs(available_phases_b_1-theta_opt));
        [~,index_b_2] = min(abs(available_phases_b_2-theta_opt));
        closest_theta_b_1(n) = available_phases_b_1(index_b_1);
        closest_theta_b_2(n) = available_phases_b_2(index_b_2);
    end
    Phi_b_1 = diag(exp(-1i*closest_theta_b_1));
    Phi_b_2 = diag(exp(-1i*closest_theta_b_2));
    instantaneous_rates_b_1(i) = log2(1+abs(h_2(1,:,i)*Phi_b_1*h_1(:,1,i))^2);
    instantaneous_rates_b_2(i) = log2(1+abs(h_2(1,:,i)*Phi_b_2*h_1(:,1,i))^2);
end
Rate_b_1_est_prop(snr) = mean(instantaneous_rates_b_1);
Rate_b_2_est_prop(snr) = mean(instantaneous_rates_b_2);

end

figure;
p=plot(snr_db_range, Rate_b_1);
set(p,'LineWidth',1, 'LineStyle', '-', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'White', 'Marker', 's', 'MarkerSize', 6, 'Color', 'Black');
hold on;
p=plot(snr_db_range, Rate_b_2);
set(p,'LineWidth',1, 'LineStyle', '-', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'Black', 'Marker', 'h', 'MarkerSize', 6, 'Color', 'Black');
hold on;
p=plot(snr_db_range, Rate_b_1_est_conv);
set(p,'LineWidth',1, 'LineStyle', '--', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'White', 'Marker', 's', 'MarkerSize', 6, 'Color', 'Black');
hold on;
p=plot(snr_db_range, Rate_b_2_est_conv);
set(p,'LineWidth',1, 'LineStyle', '--', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'Black', 'Marker', 'h', 'MarkerSize', 6, 'Color', 'Black');
hold on;
p=plot(snr_db_range, Rate_b_1_est_prop);
set(p,'LineWidth',1, 'LineStyle', '-.', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'White', 'Marker', 's', 'MarkerSize', 6, 'Color', 'Black');
hold on;
p=plot(snr_db_range, Rate_b_2_est_prop);
set(p,'LineWidth',1, 'LineStyle', '-.', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'Black', 'Marker', 'h', 'MarkerSize', 6, 'Color', 'Black');




grid on;
xlabel('SNR (dB)')
ylabel('Rate (bit/sec)');
legend('Perfect CSI, b=1','Perfect CSI, b=2', 'Estimated CSI - OMP-MMV, b=1', 'Estimated CSI - OMP-MMV, b=2', 'Estimated CSI - Proposed, b=1','Estimated CSI - Proposed, b=2')