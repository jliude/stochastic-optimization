
%% svmguide3 l_1 = 1e-3 l_2 = 1e-3
subplot(2, 2, 1)
opt = 0.49920407;

hold on
% p(1) = semilogy(infos1_prox2_saga_001.iter(1: 5: 500) * 200 /1243, infos1_prox2_saga_001.cost(1: 5: 500) - opt, 'g--');
% hold on
p(2) = semilogy(infos1_prox2_saga_01.iter(1: 5: 500)* 200 /1243, infos1_prox2_saga_01.cost(1: 5: 500) - opt, 'k-', 'LineWidth', 1);
hold on
p(3) = semilogy(infos1_prox2_saga_5.iter(1: 5: 500)* 200 /1243, infos1_prox2_saga_5.cost(1: 5: 500) - opt, 'b:', 'LineWidth', 1);
hold on
p(4) = semilogy(infos1_saga.iter(1: 5: 500)* 200 /1243, infos1_saga.cost(1: 5: 500) - opt, 'm-.', 'LineWidth', 1);
hold on
p(5) = semilogy(infos1_sdca.iter(1: 5: 500)* 200 /1243, infos1_sdca.cost(1: 5: 500) - opt, '-', 'Color', [1 0.5 0], 'LineWidth', 1);
hold on
p(6) = semilogy(infos1_sgd.iter(1: 5: 500)* 200 /1243, infos1_sgd.cost(1: 5: 500) - opt, 'c--', 'LineWidth', 1);
% hold off

legend(p(2:3), {'Prox2-SAGA, \gamma=0.1',...
    'Prox2-SAGA, \gamma=5'}, 'FontSize', 7)
% legend boxoff
ah = axes('position', get(gca,'position'), 'visible', 'off');
legend(ah, p(4:6), {'Prox-SAGA', 'Prox-SDCA ', ' Prox-SGD'}, 'FontSize', 7)
% legend boxoff


%% rcv l_1 = 1e-5 l_2 = 1e-5
subplot(2, 2, 2)
opt = 0.073940;

% p(1) = semilogy(infos2_prox2_saga_001.iter(1: 200)*8000/20242, infos2_prox2_saga_001.cost(1: 200) - opt, 'g--');
hold on
p(2) = semilogy(infos2_prox2_saga_05.iter(1: 200)*8000/20242, infos2_prox2_saga_05.cost(1: 200) - opt, 'k-', 'LineWidth', 1);
hold on
p(3) = semilogy(infos2_prox2_saga_50.iter(1: 200)*8000/20242, infos2_prox2_saga_50.cost(1: 200) - opt, 'b:', 'LineWidth', 1);
hold on
p(4) = semilogy(infos2_saga.iter(1: 200)*8000/20242, infos2_saga.cost(1: 200) - opt, 'm-.', 'LineWidth', 1);
hold on
p(5) = semilogy(infos2_sdca.iter(1: 200)*8000/20242, infos2_sdca.cost(1: 200) - opt, '-', 'Color', [1 0.5 0], 'LineWidth', 1);
hold on
p(6) = semilogy(infos2_sgd.iter(1: 200)*8000/20242, infos2_sgd.cost(1: 200) - opt, 'c--', 'LineWidth', 1);
% hold off

legend(p(2:3), {'Prox2-SAGA, \gamma=0.5',...
    'Prox2-SAGA, \gamma=50'}, 'FontSize', 7)
% legend boxoff
ah = axes('position', get(gca,'position'), 'visible', 'off');
legend(ah, p(4:6), {'Prox-SAGA', 'Prox-SDCA ', ' Prox-SGD'}, 'FontSize', 7)
% legend boxoff

%% covtype l_1 = 1e-5 l_2 = -5
subplot(2, 2, 3)
opt = 0.5831547;

% p(1) = semilogy(infos3_prox2_saga_0001.iter(1: 10: 500)*20000/581012, infos3_prox2_saga_0001.cost(1: 10: 500) - opt, 'g--');
hold on
p(2) = semilogy(infos3_prox2_saga_005.iter(1: 10: 500)*20000/581012, infos3_prox2_saga_005.cost(1: 10: 500) - opt, 'k-', 'LineWidth', 1);
hold on
p(3) = semilogy(infos3_prox2_saga_1.iter(1: 10: 500)*20000/581012, infos3_prox2_saga_1.cost(1: 10: 500) - opt, 'b:', 'LineWidth', 1);
hold on
p(4) = semilogy(infos3_saga.iter(1: 10: 500)*20000/581012, infos3_saga.cost(1: 10: 500) - opt, 'm-.', 'LineWidth', 1);
hold on
p(5) = semilogy(infos3_sdca.iter(1: 10: 500)*20000/581012, infos3_sdca.cost(1: 10: 500) - opt, '-', 'Color', [1 0.5 0], 'LineWidth', 1);
hold on
p(6) = semilogy(infos3_sgd.iter(1: 10: 500)*20000/581012, infos3_sgd.cost(1: 10: 500) - opt, 'c--', 'LineWidth', 1);
% hold off

legend(p(2:3), {'Prox2-SAGA, \gamma=0.05',...
    'Prox2-SAGA, \gamma=1'}, 'FontSize', 7)
% legend boxoff
ah = axes('position', get(gca,'position'), 'visible', 'off');
legend(ah, p(4:6), {'Prox-SAGA', 'Prox-SDCA ', ' Prox-SGD'}, 'FontSize', 7)
% legend boxoff


%% ijcnn1 l_1 = 1e-4 l_2 = 1e-5
subplot(2, 2, 4)
opt = 0.1895474;
hold on
p(2) = semilogy(infos4_prox2_saga_01.iter(1: 8: 400)*2000/49990, infos4_prox2_saga_01.cost(1: 8: 400) - opt, 'k-', 'LineWidth', 1);
hold on
p(3) = semilogy(infos4_prox2_saga_10.iter(1: 8: 400)*2000/49990, infos4_prox2_saga_10.cost(1: 8: 400) - opt, 'b:', 'LineWidth', 1);
hold on
p(4) = semilogy(infos4_saga.iter(1: 8: 400)*2000/49990, infos4_saga.cost(1: 8: 400) - opt, 'm-.', 'LineWidth', 1);
hold on
p(5) = semilogy(infos4_sdca.iter(1: 8: 400)*2000/49990, infos4_sdca.cost(1: 8: 400) - opt, '-', 'Color', [1 0.5 0], 'LineWidth', 1);
hold on
p(6) = semilogy(infos4_sgd.iter(1: 8: 400)*2000/49990, infos4_sgd.cost(1: 8: 400) - opt, 'c--', 'LineWidth', 1);

legend(p(2:3), {'Prox2-SAGA, \gamma=0.1',...
    'Prox2-SAGA, \gamma=10'}, 'FontSize', 7)
% legend boxoff
ah = axes('position', get(gca,'position'), 'visible', 'off');
legend(ah, p(4:6), {'Prox-SAGA', 'Prox-SDCA ', ' Prox-SGD'}, 'FontSize', 7)




