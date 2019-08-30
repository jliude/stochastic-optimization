
%% svmguide3 l_1 = 1e-3 l_2 = 1e-3
subplot(2, 2, 1)
%opt = 0.49920407;
opt = 0.49920206;

hold on
p(1) = semilogy(infos1_DR_prox2_saga.iter(1: 5: 500)* 200 /1243, infos1_DR_prox2_saga.cost(1: 5: 500) - opt, 'k-', 'LineWidth', 1);
hold on
p(2) = semilogy(infos1_saga.iter(1: 5: 500)* 200 /1243, infos1_saga.cost(1: 5: 500) - opt, 'm-.', 'LineWidth', 1);
hold on
p(3) = semilogy(infos1_sdca.iter(1: 5: 500)* 200 /1243, infos1_sdca.cost(1: 5: 500) - opt, '-', 'Color', [1 0.5 0], 'LineWidth', 1);
hold on
p(4) = semilogy(infos1_sgd.iter(1: 5: 500)* 200 /1243, infos1_sgd.cost(1: 5: 500) - opt, 'c--', 'LineWidth', 1);
% hold off

legend(p(1:2), {'Prox2-SAGA','Prox-SAGA,'}, 'FontSize', 7)
% legend boxoff
ah = axes('position', get(gca,'position'), 'visible', 'off');
legend(ah, p(3:4), {'Prox-SDCA ', ' Prox-SGD'}, 'FontSize', 7)
% legend boxoff


%% rcv l_1 = 1e-5 l_2 = 1e-5
subplot(2, 2, 2)
opt = 0.0739405;

p(1) = semilogy(infos2_DR_prox2_saga.iter(1: 200)*8000/20242, infos2_DR_prox2_saga.cost(1: 200) - opt, 'k-', 'LineWidth', 1);
hold on
p(2) = semilogy(infos2_saga.iter(1: 200)*8000/20242, infos2_saga.cost(1: 200) - opt, 'm-.', 'LineWidth', 1);
hold on
p(3) = semilogy(infos2_sdca.iter(1: 200)*8000/20242, infos2_sdca.cost(1: 200) - opt, '-', 'Color', [1 0.5 0], 'LineWidth', 1);
hold on
p(4) = semilogy(infos2_sgd.iter(1: 200)*8000/20242, infos2_sgd.cost(1: 200) - opt, 'c--', 'LineWidth', 1);
% hold off

legend(p(1:2), {'Prox2-SAGA','Prox-SAGA,'}, 'FontSize', 7)
% legend boxoff
ah = axes('position', get(gca,'position'), 'visible', 'off');
legend(ah, p(3:4), {'Prox-SDCA ', ' Prox-SGD'}, 'FontSize', 7)
% legend boxoff


%% covtype l_1 = 1e-5 l_2 = -5
subplot(2, 2, 3)
opt = 0.5831547;

p(1) = semilogy(infos3_DR_prox2_saga.iter(1: 10: 500)*20000/581012, infos3_DR_prox2_saga.cost(1: 10: 500) - opt, 'k', 'LineWidth', 1);
hold on
p(2) = semilogy(infos3_saga.iter(1: 10: 500)*20000/581012, infos3_saga.cost(1: 10: 500) - opt, 'm-.', 'LineWidth', 1);
hold on
p(3) = semilogy(infos3_sdca.iter(1: 10: 500)*20000/581012, infos3_sdca.cost(1: 10: 500) - opt, '-', 'Color', [1 0.5 0], 'LineWidth', 1);
hold on
p(4) = semilogy(infos3_sgd.iter(1: 10: 500)*20000/581012, infos3_sgd.cost(1: 10: 500) - opt, 'c--', 'LineWidth', 1);
% hold off

legend(p(1:2), {'Prox2-SAGA','Prox-SAGA,'}, 'FontSize', 7)
% legend boxoff
ah = axes('position', get(gca,'position'), 'visible', 'off');
legend(ah, p(3:4), {'Prox-SDCA ', ' Prox-SGD'}, 'FontSize', 7)
% legend boxoff


%% ijcnn1 l_1 = 1e-4 l_2 = 1e-5
subplot(2, 2, 4)
opt = 0.1895474;
hold on

p(1) = semilogy(infos_DR_prox2_saga.iter(1: 8: 400)*2000/49990, infos_DR_prox2_saga.cost(1: 8: 400) - opt, 'k-', 'LineWidth', 1);
hold on
p(2) = semilogy(infos_saga.iter(1: 8: 400)*2000/49990, infos_saga.cost(1: 8: 400) - opt, 'm-.', 'LineWidth', 1);
hold on
p(3) = semilogy(infos_sdca.iter(1: 8: 400)*2000/49990, infos_sdca.cost(1: 8: 400) - opt, '-', 'Color', [1 0.5 0], 'LineWidth', 1);
hold on
p(4) = semilogy(infos_sgd.iter(1: 8: 400)*2000/49990, infos_sgd.cost(1: 8: 400) - opt, 'c--', 'LineWidth', 1);

legend(p(1:2), {'Prox2-SAGA','Prox-SAGA,'}, 'FontSize', 7)
% legend boxoff
ah = axes('position', get(gca,'position'), 'visible', 'off');
legend(ah, p(3:4), {'Prox-SDCA ', ' Prox-SGD'}, 'FontSize', 7)




