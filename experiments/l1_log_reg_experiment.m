% an experiment for logistic regression

clc;
close all;

%% select a dataset
disp('======== 1:mushrooms 2:covtype 3:australian 4:w8a =========')
dataset = input('Please select a datasets:');

%% iterations
switch(dataset)
        
    case 1
        %% prepare data
        [y_train, x_train] = libsvmread('D:\cluster-svrg\data\mushrooms');
        
        x_train = x_train';
        y_train = y_train';
        y_train = y_train *2 -3;
                
        lambda1 = 1e-4;
        lambda2 = 0;
        
    case 2
        %% prepare data
        [y_train, x_train] = libsvmread('D:\Stochastic optimization\data\logistic_regression\covtype');
        x_train = x_train';
        y_train = y_train';
        y_train = (y_train - 1) * 2 - 1;
        lambda1 = 1e-6;
        lambda2 = 1e-7;
        
    case 3
        %% prepare data
        [y_train, x_train] = libsvmread('D:\Stochastic optimization\data\australian_scale');
        x_train = x_train';
        y_train = y_train';
        lambda1 = 1e-3;
        lambda2 = 1e-3;
        
    case 4
        %% prepare data
        [y_train, x_train] = libsvmread('D:\cluster-svrg\data\w8a\w8a');
%         [y_test, x_test] = libsvmread('D:\cluster-svrg\data\w8a\w8a_t');
        
        x_train = x_train';
        y_train = y_train';
        lambda1 = 2 * 1e-5;
        lambda2 = 2 * 1e-7;
        
    case 5
        [y_train, x_train] = libsvmread('D:\cluster-svrg\data\a9a');
        
        x_train = x_train';
        y_train = y_train';
        lambda1 = 3 * 1e-5;
        lambda2 = 3 * 1e-6;
        
    case 6
        [y_train, x_train] = libsvmread('D:\cluster-svrg\data\w7a');
        
        x_train = x_train';
        y_train = y_train';
        lambda1 = 5 * 1e-5;
        lambda2 = 0;
        
        
    otherwise
        disp('ERROE!')
        
end


%% define problem definition
problem = logistic_regression(x_train, y_train, 0, 0, lambda1, lambda2);


%% Calculate solution
%         disp('Solution: ');
%         w_opt = problem.calc_solution(problem, 5000);
%         disp('Min of f: ');
%         f_opt = problem.cost(w_opt);

w_opt = -inf;
f_opt = -inf;
%f_opt = 0.0199243380514867;
%f_opt = 0.008854911;

% %% perform mig
% disp('=================== mig ====================')
% loc_options = struct('f_opt', f_opt, 'w_opt', w_opt, 'max_epoch', 40, 'stepsize', 0.5);
% [~, infos_mig] = mig(problem, loc_options);

%% perform point-saga
% disp('=================== point-saga ====================')
% loc_options = struct('f_opt', f_opt, 'w_opt', w_opt, 'max_epoch', 300, 'stepsize', 0.05);
% [~, infos_prox2_saga_01] = point_saga(problem, loc_options);

%% perform point-saga
% disp('=================== Katyusha ====================')
% loc_options = struct('f_opt', f_opt, 'w_opt', w_opt, 'max_epoch', 100);
% [~, infos_Katyusha] = Katyusha(problem, loc_options);

%% perform solver
% disp('=================== SGD ====================')
% loc_options = struct('f_opt', f_opt, 'max_epoch', 400, 'stepsize', 1, 'step_alg', 'decay', 'lambda', 0.0001);
% [~, infos_sgd] = sgd(problem, loc_options);

%% perform sdca
% disp('=================== sdca ====================')
% loc_options = struct('f_opt', f_opt, 'w_opt', w_opt, 'max_epoch', 350);
% [~, infos_sdca] = sdca(problem, loc_options);

%% perform acc-sdca
% disp('=================== acc-sdca ====================')
% loc_options = struct('f_opt', f_opt, 'w_opt', w_opt, 'max_epoch', 400);
% [~, infos_acc_sdca] = accelerated_prox_sdca(problem, loc_options);

% %% perform DR_prox2-saga
% disp('=================== DR_prox2-saga ====================')
% loc_options = struct('f_opt', f_opt, 'w_opt', w_opt, 'max_epoch', 350, 'stepsize', 1);
% [~, infos_DR_prox2_saga] = DR_prox2_saga(problem, loc_options);

% 
% % 
%% perform prox2-saga
% disp('=================== prox2-saga ====================')
% loc_options = struct('f_opt', f_opt, 'w_opt', w_opt, 'max_epoch', 400, 'stepsize', 0.1);
% [~, infos_prox2_saga_01] = prox2_saga(problem, loc_options);
% 
% % perform saga
% disp('=================== saga ====================')
% loc_options = struct('f_opt', f_opt, 'w_opt', w_opt, 'max_epoch', 350, 'stepsize', 0.15);
% [~, infos_saga] = saga2(problem, loc_options);

