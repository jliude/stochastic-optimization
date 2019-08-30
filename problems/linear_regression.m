function Problem = linear_regression(x_train, y_train, x_test, y_test, lambda)
% This file defines l2-regularized linear regression problem
%
% Inputs:
%       x_train     train data matrix of x of size dxn.
%       y_train     train data vector of y of size 1xn.
%       x_test      test data matrix of x of size dxn.
%       y_test      test data vector of y of size 1xn.
%       lambda      l2-regularized parameter. 
%       stepsize    extra variable for prox_loss
% Output:
%       Problem     problem instance. 
%
%
% The problem of interest is defined as
%
%           min f(w) = 1/n * sum_i^n f_i(w),           
%           where 
%           f_i(w) = 1/2 * (w' * x_i - y_i)^2 + lambda/2 * w^2.
%
% "w" is the model parameter of size d vector.

    d = size(x_train, 1);
    n_train = length(y_train);
    n_test = length(y_test);
    
    % for prox_loss
    x_norm = sum( x_train.^2 );   

    Problem.name = @() 'linear_regression';    
    Problem.dim = @() d;
    Problem.samples = @() n_train;
    Problem.lambda = @() lambda;    
    Problem.hessain_w_independent = @() true;
    Problem.x_norm = @() sum(x_train.^2,1);
    Problem.x = @() x_train;  
    
    Problem.cost = @cost;
    function f = cost(w)
        f = sum((w'*x_train - y_train).^2)/ (2 * n_train) + lambda/2 *(w'*w);
    end

    Problem.grad = @grad;
    function g = grad(w, indices)

        residual = w'*x_train(:,indices)-y_train(indices);
        g = x_train(:,indices) * residual'/length(indices)+ lambda*w;
        
    end

    Problem.full_grad = @full_grad;
    function g = full_grad(w)

        g = grad(w, 1:n_train);
        
    end

    Problem.ind_grad = @ind_grad;
    function g = ind_grad(w, indices)
 
        residual = w'*x_train(:,indices)-y_train(indices);
        g = x_train(:,indices) * diag(residual) + lambda* repmat(w, [1 length(indices)]);
         
    end

    % for point-saga\svrg algorithm
    Problem.prox_loss = @prox_loss;
    % z: variable    t: stepsize   j: idx
    function s = prox_loss(z, j, t)
        lambda_t = lambda * t;
        mu = 1 - lambda_t / (1 + lambda_t);
        mu_t = mu*t;   mu_z = mu*z;
        x_j_norm = x_norm(j);
        
        t_bar = mu_t * x_j_norm;
        a = mu_z' * x_train(:, j);
        c = ( a + t_bar * y_train(j)) / (1 + t_bar);
        s = mu_z - (a - c) * x_train(:, j) / x_j_norm;
    end

% 
%     Problem.hess = @hess; 
%     function h = hess(w, indices)
% %         % original code
% %         h = 0;
% %         len = length(indices);
% %         for ii=1:len
% %             idx = indices(ii);
% %             xx = x_train(:,indices(:,idx));
% %             h = h + xx * xx';
% %         end
% %         h = h/len + lambda * eye(d);
%         
%         h = 1/length(indices) * x_train(:,indices) * (x_train(:,indices)') + lambda * eye(d);
%     end

%     Problem.full_hess = @full_hess; 
%     function h = full_hess(w)
%         
%         h = hess(w, 1:n_train);
%         
%     end
% 
%     Problem.hess_vec = @hess_vec; 
%     function hv = hess_vec(w, v, indices)
%         
%         hv = 1/length(indices) * x_train(:,indices) * ((x_train(:,indices)'*v)) + lambda*v;
%         
%     end

    Problem.prediction = @prediction;
    function p = prediction(w)
        p = w' * x_test;        
    end

    Problem.mse = @mse;
    function e = mse(y_pred)
        
        e = sum((y_pred-y_test).^2)/ (2 * n_test);
        
    end

    Problem.calc_solution = @calc_solution;
    function w_opt = calc_solution(problem, maxiter, method)
        
        if nargin < 3
            method = 'lbfgs';
        end        
        
        options.max_iter = maxiter;
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
        else  
            options.step_alg = 'backtracking';  
            options.mem_size = 5;
            [w_opt,~] = lbfgs(problem, options);              
        end
    end

end