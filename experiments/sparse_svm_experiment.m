
% an experiment for logistic regression

clc;
clear;
close all;

%% select a dataset
disp('======== 1:ijcnn, 2:covtype 3:leu 4:rcv 5: real-sim 6:gisette_scale 7:mushroom=========')
dataset = input('Please select a datasets:');

%% iterations
switch(dataset)
    case 1
        %% prepare data
        [y_train, x_train] = libsvmread('D:\Stochastic optimization\data\logistic_regression\ijcnn1\ijcnn1.tr');
        [y_test, x_test] = libsvmread('D:\Stochastic optimization\data\logistic_regression\ijcnn1\ijcnn1.t');
        x_train = x_train';
        x_test = x_test';
        y_train = y_train';
        y_test = y_test';
        lambda1 = 1e-4;
        lambda2 = 1e-5;
        
    case 2
        %% prepare data
        [y_train, x_train] = libsvmread('D:\Stochastic optimization\data\logistic_regression\covtype');
        x_train = x_train';
        y_train = y_train';
        y_train = (y_train - 1) * 2 - 1;
        lambda1 = 1e-5;
        lambda2 = 1e-5;
        
        
    case 3
        %% prepare data
        [y_train, x_train] = libsvmread('D:\Stochastic optimization\data\logistic_regression\leu');
        x_train = x_train';
        y_train = y_train';
%         y_train = (y_train - 1) * 2 - 1;
        lambda1 = 0.01;
        lambda2 = 0.01;
        
    case 4
        %% prepare data
        [y_train, x_train] = libsvmread('D:\Stochastic optimization\data\logistic_regression\rcv1');
        x_train = x_train';
        y_train = y_train';
        lambda1 = 1e-5;
        lambda2 = 1e-5;
        
        
    case 5
        %% prepare data
        [y_train, x_train] = libsvmread('D:\Stochastic optimization\data\logistic_regression\real-sim');
        x_train = x_train';
        y_train = y_train';
        y_train = (y_train - 1) * 2 - 1;
        lambda1 = 0.0001;
        lambda2 = 0.0001;
        
    case 6
        %% prepare data
        [y_train, x_train] = libsvmread('D:\Stochastic optimization\data\logistic_regression\gisette_scale');
        x_train = x_train';
        y_train = y_train';
        lambda1 = 0.00001;
        lambda2 = 0.000001;
        
%         lambda1 = 0;
%         lambda2 = 0;

    case 7
        %% prepare data
        [y_train, x_train] = libsvmread('D:\cluster-svrg\data\mushrooms');
        
        x_train = x_train';
        y_train = y_train';
        y_train = y_train *2 -3;
        
        x_test = 0;y_test=0;
        
        lambda1 = 0.;
        lambda2 = 0.001;
        
    case 8
        %% prepare data
        [y_train, x_train] = libsvmread('D:\cluster-svrg\data\fourclass_scale');
        
        x_train = x_train';
        y_train = y_train';
        
        x_test = 0;y_test=0;
        
        lambda1 = 0.01;
        lambda2 = 0.01;
        
    case 9
        [y_train, x_train] = libsvmread('D:\cluster-svrg\data\svmguide3');
                
        x_train = x_train';
        y_train = y_train';
        
        lambda1 = 1e-3;
        lambda2 = 1e-3;
        
    otherwise
        disp('ERROE!')
        
end

        
%% Options for optimization algorithms

%% define problem definition
problem = sparse_svm(x_train, y_train, 0, 0, lambda1, lambda2);


%% Calculate solution
%         disp('Solution: ');
%         w_opt = problem.calc_solution(problem, 5000);
%         disp('Min of f: ');
%         f_opt = problem.cost(w_opt);
f_opt = -inf; 
w_opt = -inf;

% 
%% perform DR_prox2-saga
disp('=================== DR_prox2-saga ====================')
loc_options = struct('f_opt', f_opt, 'w_opt', w_opt, 'max_epoch', 500, 'stepsize', 1);
[~, infos_DR_prox2_saga] = DR_prox2_saga(problem, loc_options);

%% perform prox2-saga
% disp('=================== prox2-saga ====================')
% loc_options = struct('f_opt', f_opt, 'w_opt', w_opt, 'max_epoch', 500, 'stepsize', 0.0001);
% [~, infos_prox2_saga_001] = prox2_saga(problem, loc_options);

% %% perform accelerated_prox_sdca
% disp('=================== Accelerated Prox SDCA ====================')
% loc_options = struct('f_opt', f_opt, 'w_opt', w_opt, 'max_epoch', 500);
% [~, infos_accelerated_prox_sdca] = accelerated_prox_sdca(problem, loc_options);

%% perform SDCA
% disp('=================== SDCA ====================')
% loc_options = struct('f_opt', f_opt, 'w_opt', w_opt, 'max_epoch', 500);
% [~, infos_sdca] = sdca(problem, loc_options);

% %% perform Pegasos
% disp('=================== Pegasos ====================')
% loc_options = struct('f_opt', f_opt, 'w_opt', w_opt, 'max_epoch', 500, 'step_alg', 'decay', 'lambda', 0.0001);
% [~, infos_sgd] = sgd(problem, loc_options);

% %% perform point-saga
% disp('=================== Point-saga ====================')
% loc_options = struct('f_opt', f_opt, 'w_opt', w_opt, 'max_epoch', 50, 'stepsize', 0.1);
% [~, infos_point_saga] = point_saga(problem, loc_options);

%% perform s-ppg
% disp('=================== s_ppg ====================')
% loc_options = struct('f_opt', f_opt, 'w_opt', w_opt, 'max_epoch', 100, 'stepsize', 10);
% [~, infos_s_ppg] = s_ppg(problem, loc_options);

%% perform jc-saga
% disp('=================== jc-saga ====================')
% loc_options = struct('f_opt', f_opt, 'w_opt', w_opt, 'max_epoch', 1000, 'stepsize', 0.01);
% [~, infos_jc_saga] = jc_saga(problem, loc_options);
% 
%% perform Pegasos
% disp('=================== Pegasos ====================')
% loc_options = struct('f_opt', f_opt, 'w_opt', w_opt, 'max_epoch', 100, 'step_alg', 'decay-4', 'lambda', 1);
% [~, infos_sgd] = sgd(problem, loc_options);

% 
%% perform saga
% disp('=================== SAGA ====================')
% loc_options = struct('f_opt', f_opt, 'max_epoch', 500, 'stepsize', 0.1);
% [~, infos_saga] = saga2(problem, loc_options);


%% perform point-saga-plus
% disp('=================== Point-saga-plus ====================')
% loc_options = struct('f_opt', f_opt, 'max_epoch', 100);
% [~, infos_point_saga_plus] = point_saga_plus(problem, loc_options);
% 
% %% perform svrg
% disp('=================== SVRG ====================')
% loc_options = struct('f_opt', f_opt, 'stepsize', 0.001);
% [~, infos_svrg] = svrg(problem, loc_options);
% 
% %% perform sag
% disp('=================== SAG ====================')
% loc_options = struct('f_opt', f_opt);
% [~, infos_point_sag] = sag(problem, loc_options);

