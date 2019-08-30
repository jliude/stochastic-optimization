
% an experiment for logistic regression

clc;
clear;
close all;

%% select a dataset
disp('======== 1:ijcnn, 2:covtype =========')
dataset = input('Please select a datasets:')

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
        lambda = 0.0000001;
        
        %% define problem definition
        problem = logistic_regression(x_train, y_train, x_test, y_test, lambda);
        
        %% Options for optimization algorithms
        
        %% Calculate solution
        disp('Solution: ');
        w_opt = problem.calc_solution(problem, 1000);
        disp('Min of f: ');
        f_opt = problem.cost(w_opt);
        
        % %% perform sgd
        % disp('=================== SGD ====================')
        % loc_options = struct('f_opt', f_opt);
        % [~, ~] = sgd(problem, loc_options);
                
                        
        %% perform saga
        disp('=================== SAGA ====================')
        loc_options = struct('f_opt', f_opt, 'stepsize', 0.1);
        [~, ~] = saga(problem, loc_options);
        
        %% perform point-saga
        disp('=================== Point-saga ====================')
        loc_options = struct('f_opt', f_opt, 'stepsize', 0.1);
        [~, ~] = point_saga(problem, loc_options);

        %% perform svrg
        disp('=================== SVRG ====================')
        loc_options = struct('f_opt', f_opt);
        [~, ~] = svrg(problem, loc_options);
        
        %% perform sag
        disp('=================== SAG ====================')
        loc_options = struct('sub_mode', 'SAG', 'f_opt', f_opt);
        [~, ~] = sag(problem, loc_options);
        
        
    case 2
        %% prepare data
        [y_train, x_train] = libsvmread('D:\Stochastic optimization\data\logistic_regression\covtype');
        x_train = x_train';
        y_train = y_train';
        y_train = (y_train - 1) * 2 - 1;
        lambda = 0.00001;
        
        %% define problem definition
        problem = logistic_regression(x_train, y_train, 0, 0, lambda);
        
        %% Options for optimization algorithms
        
        %% Calculate solution
        disp('Solution: ');
        w_opt = problem.calc_solution(problem, 100);
        disp('Min of f: ');
        f_opt = problem.cost(w_opt)
        
        %% perform sgd
        disp('=================== SGD ====================')
        loc_options = struct('f_opt', f_opt);
        [~, ~] = sgd(problem, loc_options); 
                        
%         %% perform saga
%         disp('=================== SAGA ====================')
%         loc_options = struct('sub_mode', 'SAGA', 'f_opt', f_opt);
%         [~, ~] = saga(problem, loc_options);
%         
%         %% perform svrg
%         disp('=================== SVRG ====================')
%         loc_options = struct('f_opt', f_opt);
%         [~, ~] = svrg(problem, loc_options);
%         
%         %% perform sag
%         disp('=================== SAG ====================')
%         loc_options = struct('sub_mode', 'SAG', 'f_opt', f_opt);
%         [~, ~] = sag(problem, loc_options);
        
    otherwise 
        disp('ERROE!')        
end
