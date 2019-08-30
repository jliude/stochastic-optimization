% an experiment for linear regression

clc;
clear;
close all;

%% prepare datasets
[y_train, x_train] = libsvmread('D:\Stochastic optimization\data\linear_regression\cpusmall');
% [y_test, x_test] = libsvmread('D:\Stochastic optimization\data\linear_regression\YearPredictionMSD.t');
x_train = x_train';
% x_test = x_test';
y_train = y_train';
% y_test = y_test';
lambda = 0.00001;

%% define problem definition
problem = linear_regression(x_train, y_train, 0, 0, lambda);

%% Options for optimization algorithms

%% Calculate solution
disp('Solution: ');
w_opt = problem.calc_solution(problem, 10000);
disp('Min of f: ');
f_opt = problem.cost(w_opt)

% %% perform sgd
% disp('=================== SGD ====================')
% loc_options = struct('f_opt', f_opt);
% [~, ~] = sgd(problem, loc_options);

%% perform saga
disp('=================== SAGA ====================')
loc_options = struct('sub_mode', 'SAGA', 'f_opt', f_opt, 'max_epoch', 1000, 'stepsize', 0.01);
[~, ~] = saga(problem, loc_options);

%% perform point-saga
disp('=================== Point-SAGA ====================')
loc_options = struct('f_opt', f_opt);
[~, ~] = point_saga(problem, loc_options);

%% perform svrg
disp('=================== SVRG ====================')
loc_options = struct('f_opt', f_opt);
[~, ~] = svrg(problem, loc_options);

%% perform sag
disp('=================== SAG ====================')
loc_options = struct('sub_mode', 'SAG', 'f_opt', f_opt);
[~, ~] = sag(problem, loc_options);