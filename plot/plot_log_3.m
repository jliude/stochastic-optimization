
%% mushrooms lambda1 = 1e-4
% lambda2 = 1e-4
subplot(4, 2, 1)

% opt = 0.01992434127;
opt = 0.01992433805148647;

semilogy(infos1_DR_prox2_saga_4.iter(1: 3: 350)*500/8124, infos1_DR_prox2_saga_4.cost(1: 3: 350) - opt, 'k-', 'LineWidth', 1 )
hold on

semilogy(infos1_sdca_4.iter(1: 3: 350)*500/8124, infos1_sdca_4.cost(1: 3: 350) - opt, 'm-.', 'LineWidth', 1)
hold on
semilogy(infos1_acc_sdca_4.iter(1: 3: 350)*500/8124, infos1_acc_sdca_4.cost(1: 3: 350) - opt, '-', 'Color', [1 0.5 0], 'LineWidth', 1)
hold on
semilogy(infos1_saga_4.iter(1: 3: 350)*500/8124, infos1_saga_4.cost(1: 3: 350) - opt, 'g--', 'LineWidth', 1)
hold on
semilogy(infos1_sgd_4.iter(1 :3: 350)*500/8124, infos1_sgd_4.cost(1: 3: 350) - opt, 'b:', 'LineWidth', 1)

axis([0 1.7*1e5/8124 0 1])
ax = gca;
ax.FontSize = 8;
ax.YTick = [10^(-9) 10^(-8) 10^(-7) 10^(-6) 10^(-5) 10^(-4) 10^(-3) 10^(-2) 10^(-1) 1];
ylim([10^(-9) 1]);
xlabel('epoch')
ylabel('objective gap')
title('mushrooms, \lambda_2 = 10^{-4}', 'FontSize', 10)
% legend('sdca', 'acc\_sdca', 'saga', 'sgd', 'prox2\_saga')
hold off

% lambda2 = 1e-5
subplot(4, 2, 3)
% opt = 0.010782009776;
opt = 0.01078200944;

semilogy(infos1_DR_prox2_saga_5.iter(1: 3: 200)*2000/8124, infos1_DR_prox2_saga_5.cost(1: 3: 200) - opt, 'k-', 'LineWidth', 1)
hold on

semilogy(infos1_sdca_5.iter(1: 3: 200)*2000/8124, infos1_sdca_5.cost(1: 3: 200) - opt, 'm-.', 'LineWidth', 1)
hold on
semilogy(infos1_acc_sdca_5.iter(1: 3: 200)*2000/8124, infos1_acc_sdca_5.cost(1: 3: 200) - opt, '-', 'Color', [1 0.5 0], 'LineWidth', 1)
hold on
semilogy(infos1_saga_5.iter(1: 3: 200)*2000/8124, infos1_saga_5.cost(1: 3: 200) - opt, 'g--', 'LineWidth', 1)
hold on
semilogy( infos1_sgd_5.iter(1 :3: 200)*2000/8124, infos1_sgd_5.cost(1: 3: 200) - opt, 'b:', 'LineWidth', 1)

ax = gca;
ax.FontSize = 8;
ax.YTick = [10^(-9) 10^(-8) 10^(-7) 10^(-6) 10^(-5) 10^(-4) 10^(-3) 10^(-2) 10^(-1) 1];
xlim([0 49])
ylim([10^(-10) 1])
xlabel('epoch')
ylabel('objective gap')
title('mushrooms, \lambda_2 = 10^{-5}', 'FontSize', 10)
% legend('sdca', 'acc\_sdca', 'saga', 'sgd', 'prox2\_saga')
hold off

% lambda2 = 1e-6
subplot(4, 2, 5)

opt = 0.008854911;
semilogy(infos1_DR_prox2_saga_6.iter(1: 3: 350)*2000/8124, infos1_DR_prox2_saga_6.cost(1: 3: 350) - opt, 'k-', 'LineWidth', 1)
hold on
opt = 0.008854911;
semilogy(infos1_sdca_6.iter(1: 3: 350)*2000/8124, infos1_sdca_6.cost(1: 3: 350) - opt, 'm-.', 'LineWidth', 1)
hold on
semilogy(infos1_acc_sdca_6.iter(1: 3: 350)*2000/8124, infos1_acc_sdca_6.cost(1: 3: 350) - opt, '-', 'Color', [1 0.5 0], 'LineWidth', 1)
hold on   
semilogy(infos1_saga_6.iter(1: 3: 350)*2000/8124, infos1_saga_6.cost(1: 3: 350) - opt, 'g--', 'LineWidth', 1)
hold on
semilogy(infos1_sgd_6.iter(1 :3: 350)*2000/8124, infos1_sgd_6.cost(1: 3: 350) - opt, 'b:', 'LineWidth', 1)

axis([0 7* 1e5/8124 0 1])
ax = gca;
ax.FontSize = 8;
ax.YTick = [10^(-9) 10^(-8) 10^(-7) 10^(-6) 10^(-5) 10^(-4) 10^(-3) 10^(-2) 10^(-1) 1];
ylim([10^(-9) 1])
xlabel('epoch')
ylabel('objective gap')
title('mushrooms, \lambda_2 = 10^{-6}', 'FontSize', 10)
% legend('sdca', 'acc\_sdca', 'saga', 'sgd', 'prox2\_saga')
hold off

%% w7a % lambda1 = 5e-5


% lambda2 = 1e-4
subplot(4, 2, 2)
opt = 0.149256164307275;

semilogy(infos2_DR_prox2_saga_6_5.iter(1: 150)*4000/24692, infos2_DR_prox2_saga_6_5.cost(1: 150) - opt, 'k-', 'LineWidth', 1)
hold on
semilogy(infos2_sdca_6_5.iter(1: 150)*4000/24692, infos2_sdca_6_5.cost(1: 150) - opt, 'm-.', 'LineWidth', 1)
hold on
semilogy(infos2_acc_sdca_6_5.iter(1: 150)*4000/24692, infos2_acc_sdca_6_5.cost(1: 150) - opt, '-', 'Color', [1 0.5 0], 'LineWidth', 1)
hold on   
semilogy(infos2_saga_6_5.iter(1: 150)*4000/24692, infos2_saga_6_5.cost(1: 150) - opt, 'g--', 'LineWidth', 1)
hold on
semilogy(infos2_sgd_6_5.iter(1 :150)*4000/24692, infos2_sgd_6_5.cost(1: 150) - opt, 'b:', 'LineWidth', 1)

ax = gca;
ax.FontSize = 8;
ax.YTick = [10^(-9) 10^(-8) 10^(-7) 10^(-6) 10^(-5) 10^(-4) 10^(-3) 10^(-2) 10^(-1) 1];
xlim([0 18])
ylim([10^(-10) 1])
xlabel('epoch')
ylabel('objective gap')
title('w7a, \lambda_2=10^{-4}', 'FontSize', 10)
hold off

% subplot(2, 3, 5)
% opt = 0.12953648;
% 
% semilogy(infos2_sdca_7.iter(1: 3: 350)*4000, infos2_sdca_7.cost(1: 3: 350) - opt, 'm-.')
% hold on
% semilogy(infos2_acc_sdca_7.iter(1: 3: 350)*4000, infos2_acc_sdca_7.cost(1: 3: 350) - opt, 'm-')
% hold on   
% semilogy(infos2_saga_7.iter(1: 3: 350)*4000, infos2_saga_7.cost(1: 3: 350) - opt, 'b:')
% hold on
% semilogy(infos2_prox2_saga_7.iter(1: 3: 350)*4000, infos2_prox2_saga_7.cost(1: 3: 350) - opt, 'k-')
% hold on
% semilogy(infos2_sgd_7.iter(1 :3: 350)*4000, infos2_sgd_7.cost(1: 3: 350) - opt, 'g--')
% 
% hold off

% lambda2 = 5*1e-5
subplot(4, 2, 4)
opt = 0.14283286407585;

line(1) = semilogy(infos2_DR_prox2_saga_5.iter(1: 200)*4000/24692, infos2_DR_prox2_saga_5.cost(1: 200) - opt, 'k-', 'LineWidth', 1);
hold on
line(2) = semilogy(infos2_sdca_5.iter(1: 200)*4000/24692, infos2_sdca_5.cost(1: 200) - opt, 'm-.', 'LineWidth', 1);
hold on
line(3) = semilogy(infos2_acc_sdca_5.iter(1: 200)*4000/24692, infos2_acc_sdca_5.cost(1: 200) - opt, '-', 'Color', [1 0.5 0], 'LineWidth', 1);
hold on   
line(4) = semilogy(infos2_saga_5.iter(1: 200)*4000/24692, infos2_saga_5.cost(1: 200) - opt, 'g--', 'LineWidth', 1);
hold on
line(5) = semilogy(infos2_sgd_5.iter(1 :200)*4000/24692, infos2_sgd_5.cost(1: 200) - opt, 'b:', 'LineWidth', 1);

axis([0 8*1e5/32561 0 1]);
ax = gca;
ax.FontSize = 8;
ax.YTick = [10^(-10) 10^(-9) 10^(-8) 10^(-7) 10^(-6) 10^(-5) 10^(-4) 10^(-3) 10^(-2) 10^(-1) 1];
ylim([10^(-11) 1])
xlabel('epoch')
ylabel('objective gap')
title('w7a, \lambda_2=5\times10^{-5}', 'FontSize', 10)
hold off

% lambda2 = 5*1e-6
subplot(4, 2, 6)
opt = 0.13188169;

semilogy(infos2_DR_prox2_saga_6.iter(1: 3: 350)*4000/24692, infos2_DR_prox2_saga_6.cost(1: 3: 350) - opt, 'k-', 'LineWidth', 1)
hold on
semilogy(infos2_sdca_6.iter(1: 3: 350)*4000/24692, infos2_sdca_6.cost(1: 3: 350) - opt, 'm-.', 'LineWidth', 1)
hold on
semilogy(infos2_acc_sdca_6.iter(1: 3: 350)*4000/24692, infos2_acc_sdca_6.cost(1: 3: 350) - opt, '-', 'Color', [1 0.5 0], 'LineWidth', 1)
hold on   
semilogy(infos2_saga_6.iter(1: 3: 350)*4000/24692, infos2_saga_6.cost(1: 3: 350) - opt, 'g--', 'LineWidth', 1)
hold on
semilogy(infos2_sgd_6.iter(1 :3: 350)*4000/24692, infos2_sgd_6.cost(1: 3: 350) - opt, 'b:', 'LineWidth', 1)


axis([0 14*1e5/32561 0 1]);
ax = gca;
ax.FontSize = 8;
ax.YTick = [ 10^(-7) 10^(-6) 10^(-5) 10^(-4) 10^(-3) 10^(-2) 10^(-1) 1];
ylim([10^(-7) 1])
xlabel('epoch')
ylabel('objective gap')
title('w7a, \lambda_2=5\times10^{-6}', 'FontSize', 10)
hold off

%% creat overall legend
subplot(4, 2, [7 8])
axis off
legend(line(1:3), {'Prox2-SAGA', 'Prox-SDCA', 'Acc-SDCA'}, 'FontSize', 8)
% legend boxoff
ah = axes('position', get(gca,'position'), 'visible', 'off');
legend(ah, line(4:5), {'Prox-SAGA', ' Prox-SGD'}, 'FontSize', 8)
% legend boxoff

% hL = legend([line1, line2, line3, line4, line5],{'Prox2-SAGA', 'Prox-SDCA',...
%     'Accelerated Prox-SDCA', 'Prox-SAGA', 'Prox-SGD'});
% newPosition = [0.5 0.45 1 1];
% newUnits = 'normalized';
% set(hL,'Position', newPosition,'Units', newUnits);