%% ijcnn l_1 = 1e-4  l_2 = 1e-5
% subplot(1, 3, 1);
% 
% opt = 0.1895474;
% semilogy(infos_prox2_saga_01.iter(1: 300)*2000, infos_prox2_saga_01.cost(1 : 300) - opt, 'k-')
% hold on
% semilogy(infos_prox2_saga_001.iter(1: 300)*2000, infos_prox2_saga_001.cost(1: 300) - opt, 'k--')
% hold on
% semilogy(infos_prox2_saga_10.iter(1: 300)*2000, infos_prox2_saga_10.cost(1: 300) - opt, 'k:')
% hold on
% % semilogy(infos_sdca.iter(1: 300)*2000, infos_sdca.cost(1: 300) - opt, 'm-')
% hold on
% semilogy(infos_accelerated_prox_sdca.iter(1: 300)*2000, infos_accelerated_prox_sdca.cost(1: 300) - opt, 'm--')
% hold on
% semilogy(infos_sgd.iter(1: 300)*2000, infos_sgd.cost(1: 300) - opt, 'm:')
% hold off
% % axis([0 12e5 10^-1.4 1])
% legend('prox2\_saga\_5', 'prox2\_saga\_001', 'prox2\_saga\_100', 'sdca', 'acc-sdca', 'sgd')


%% RCV 1_1 = 1e-5  l_2 = 1e-6
% subplot(1, 3, 2);

% opt = 0.049061;
% semilogy(infos_prox2_saga_5.iter(1: 300)*8000, infos_prox2_saga_5.cost(1: 300) - opt, 'k-')
% hold on
% semilogy(infos_prox2_saga_001.iter(1: 300)*8000, infos_prox2_saga_001.cost(1: 300) - opt, 'k--')
% hold on
% semilogy(infos_prox2_saga_100.iter(1: 300)*8000, infos_prox2_saga_100.cost(1: 300) - opt, 'k:')
% hold on
% semilogy(infos_saga.iter(1: 300)*8000, infos_saga.cost(1: 300) - opt, 'k-.')
% hold on
% semilogy(infos_sdca.iter(1: 300)*8000, infos_sdca.cost(1: 300) - opt, 'm-')
% hold on
% semilogy(infos_accelerated_prox_sdca.iter(1: 300)*8000, infos_accelerated_prox_sdca.cost(1: 300) - opt, 'm--')
% hold on
% semilogy(infos_sgd.iter(1: 300)*8000, infos_sgd.cost(1: 300) - opt, 'm:')
% hold off
% 
% % axis([0 12e5 10^-1.4 1])
% legend('prox2\_saga\_5', 'prox2\_saga\_001', 'prox2\_saga\_100', 'saga', 'sdca', 'acc-sdca')



%% covtype l_1 = 1e-5  l_2 = 1e-6
suplot(1, 3, 3)

opt = 0.582191;
semilogy(infos_prox2_saga_001.iter(1: 10: 1000), infos_prox2_saga_001.cost(1: 10: 1000) - opt, 'k-')
hold on
semilogy(infos_prox2_saga_05.iter(1: 10: 1000), infos_prox2_saga_05.cost(1: 10: 1000) - opt, 'k:')
hold on
semilogy(infos_saga.iter(1: 10: 1000), infos_saga.cost(1: 10: 1000) - opt, 'k-.')
hold on
semilogy(infos_sdca.iter(1: 10: 1000), infos_sdca.cost(1: 10: 1000) - opt, 'm-')
hold on
semilogy(infos_sgd.iter(1: 10: 1000), infos_sgd.cost(1: 10: 1000) - opt, 'm:')
%semilogy(infos_accelerated_prox_sdca.iter(1: 10: 1000), infos_accelerated_prox_sdca.cost(1: 10: 1000) - opt, 'm--')
hold off

%% svmguide3 l_1 = 1e-3  l_2 = 1e-4
subplot(1, 3, 1)

opt = 0.4900127;
semilogy(infos_prox2_saga_01.iter(1: 5: 500), infos_prox2_saga_01.cost(1: 5: 500) - opt, 'k-')
hold on
semilogy(infos_prox2_saga_001.iter(1: 5: 500), infos_prox2_saga_001.cost(1: 5: 500) - opt, 'k--')
hold on
semilogy(infos_prox2_saga_5.iter(1: 5: 500), infos_prox2_saga_5.cost(1: 5: 500) - opt, 'k:')
hold on
semilogy(infos_saga.iter(1: 5: 500), infos_saga.cost(1: 5: 500) - opt, 'k-.')
hold on
semilogy(infos_sdca.iter(1: 5: 500), infos_sdca.cost(1: 5: 500) - opt, 'm-')
hold on
semilogy(infos_sgd.iter(1: 5: 500), infos_sgd.cost(1: 5: 500) - opt, 'm:')
hold off
