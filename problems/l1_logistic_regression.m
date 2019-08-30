function [Problem] = l1_logistic_regression(x_train, y_train, x_test, y_test, varargin)
% This file defines logistic regression (binary classifier) problem with l1-norm.
%
% Inputs:
%       x_train     train data matrix of x of size dxn.
%       y_train     train data vector of y of size 1xn.
%       x_test      test data matrix of x of size dxn.
%       y_test      test data vector of y of size 1xn.
%       varargin    options.
% Output:
%       Problem     problem instance. 
%
%
% The problem of interest is defined as
%
%           min f(w) = 1/n * sum_i^n f_i(w),           
%           where 
%           f_i(w) = log(1 + exp(-y_i' .* (w'*x_i))) + lambda || w ||_1.
%
% "w" is the model parameter of size d vector.
    
    % regulation
    if nargin < 5
        lambda = 0.1;
    else
        lambda = varargin{1};
    end
    
    % dimensions and samples
    d = size(x_train, 1);
    n_train = length(y_train);
    n_test = length(y_test);
    
    Problem.name = @() 'l1 logistic_regression';
    Problem.dim = @() d;
    Problem.samples = @() n_train;
    Problem.lambda = @() lambda;
    Problem.classes = @() 2;
    Problem.hessain_w_independent = @() false;
    Problem.x_norm = @() sum(x_train.^2, 1);
    Problem.x = @() x_train;
    
    Problem.prox = @(w,t) soft_thresh(w, t * lambda);
    
    % cost
    Problem.cost = @(w) -sum(log(sigmoid(y_train.*(w'*x_train))),2)/n_train + lambda * norm(w,1);
    Problem.cost_batch = @(w, indices) -sum(log(sigmoid(y_train(indices).*(w'*x_train(:,indices))))/n_train,2) + lambda * norm(w,1);
    
    % calculate l1 norm
    Problem.reg = @(w) norm(w, 1);
    
    % gradient
    Problem.grad = @grad;
    function g = grad(w, indices)
        
        e = exp(-1*y_train(indices)'.*(x_train(:,indices)'*w));
        s = e./(1+e);
        g = -(1/length(indices))*((s.*y_train(indices)')'*x_train(:,indices)')';
        g = full(g);
        
    end

    % full gradient
    Problem.full_grad = @(w) grad(w, 1:n_train);
    
    Problem.hess = @hess;
    function h = hess(w, indices)
        sigm_val = sigmoid(y_train(indices).*(w'*x_train(:,indices)));
        c = sigm_val .* (ones(1,length(indices))-sigm_val);
        h = 1/length(indices)* x_train(:,indices) * diag(y_train(indices).^2 .* c) * x_train(:,indices)';
    end

    Problem.full_hess = @full_hess;
    function h = full_hess(w)
        
        h = hess(w, 1:n_train);
        
    end

    Problem.hess_vec = @hess_vec;
    function hv = hess_vec(w, v, indices)
        
        sigm_val = sigmoid(y_train(indices).*(w'*x_train(:,indices)));
        c = sigm_val .* (ones(1,length(indices))-sigm_val);
        hv = 1/length(indices)* x_train(:,indices) * diag(y_train(indices).^2 .* c) * (x_train(:,indices)' * v);
        
    end

    % prediction
    Problem.prediction = @prediction;
    function p = prediction(w)
                
        p = sigmoid(w' * x_test);
        
        class1_idx = p>0.5;
        class2_idx = p<=0.5;         
        p(class1_idx) = 1;
        p(class2_idx) = -1;  
        
    end

    % accuracy
    Problem.accuracy = @(y_pred) sum(y_pred == y_test) / n_test;
    
    % solution
    Problem.calc_solution = @calc_solution;
    function w_opt = calc_solution(problem, options_in, method)
        
        if nargin < 3
            method = 'gd_nesterov';
        end        
        
        options.max_iter = options_in.max_iter;
        options.w_init = options_in.w_init;
        options.verbose = true;
        options.tol_optgap = 1.0e-24;
        options.tol_gnorm = 1.0e-16;
        options.step_alg = 'backtracking';
        
        if strcmp(method, 'sg')
            [w_opt,~] = gd(problem, options);
        elseif strcmp(method, 'cg')
            [w_opt,~] = ncg(problem, options);
        elseif strcmp(method, 'newton')
            options.sub_mode = 'INEXACT';    
            options.step_alg = 'non-backtracking'; 
            [w_opt,~] = newton(problem, options);
        elseif strcmp(method, 'gd_nesterov')
            options.step_alg = 'backtracking';
            options.step_init_alg = 'bb_init';
            [w_opt,~] = gd_nesterov(problem, options);            
        else 
            options.step_alg = 'backtracking';  
            options.mem_size = 5;
            [w_opt,~] = lbfgs(problem, options);              
        end
    end
end

