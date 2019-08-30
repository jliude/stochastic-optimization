function [Problem] = logistic_regression(x_train, y_train, x_test, y_test, lambda1, lambda2)
% This file defines logistic regression (binary classifier) problem
%
% Inputs:
%       x_train     train data matrix of x of size dxn.
%       y_train     train data vector of y of size 1xn.
%       x_test      test data matrix of x of size dxn.
%       y_test      test data vector of y of size 1xn.
%       lambda      l2-regularized parameter. 
% Output:
%       Problem     problem instance. 
%
%
% The problem of interest is defined as
%
%           min f(w) = 1/n * sum_i^n f_i(w),           
%           where 
%           f_i(w) = log(1 + exp(-y_i' .* (w'*x_i))) + lambda2/2 * w^2 + lambda1*|w|.
%
% "w" is the model parameter of size d vector.
    
    % dimensions and samples number
    d = size(x_train, 1);
    n_train = length(y_train);
    n_test = length(y_test);
    
    % for prox_loss
    x_norm = sum( x_train.^2 );
    
    % defines
    Problem.name = @() 'logistic_regression';
    Problem.dim = @() d;
    Problem.samples = @() n_train;
    Problem.lambda1 = @() lambda1;
    Problem.lambda2 = @() lambda2;
    Problem.classes = @() 2;
    Problem.hessain_w_independent = @() false;
    Problem.x_norm = @() sum(x_train.^2, 1);
    Problem.x = @() x_train;
    
    Problem.cost = @(w) -sum(log(sigmoid(y_train.*(w'*x_train))),2)/n_train + lambda2 * (w'*w) / 2 + lambda1 * norm(w,1);
    Problem.cost_batch = @(w, indices) -sum(log(sigmoid(y_train(indices).*(w'*x_train(:,indices))))/n_train,2) + lambda2 * (w'*w) / 2 +  lambda1 * norm(w,1);
    
    Problem.grad = @grad;
    function g = grad(w, indices)
        e = exp(-1*y_train(indices)'.*(x_train(:,indices)'*w));
        s = e./(1+e);
        g = -(1/length(indices))*((s.*y_train(indices)')'*x_train(:,indices)')';
        g = full(g) + lambda2 * w;    
    end
    Problem.full_grad = @(w) grad(w, 1:n_train);
    
    Problem.stored_var = @store_var;
    function r = store_var(w, i)
        e = exp(-1*y_train(i)'.*(x_train(:, i)'*w));
        s = e ./ (1+e);
        r = -s.*y_train(i);
    end
    
    % get x_train(:, i)
    Problem.x_train_i = @(i) x_train(:, i);
    
    Problem.ind_grad = @ind_grad;
    function g = ind_grad(w, indices)
        g = -ones(d,1) * y_train(indices).*x_train(:,indices) * diag(ones(1,length(indices))-sigmoid(y_train(indices).*(w'*x_train(:,indices))))+ lambda2* repmat(w, [1 length(indices)]);     
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%% low storage prox2-saga %%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Problem.init = @() y_train;

    Problem.v_dot_y = @v_dot_y;
    function r = v_dot_y(z, stepsize, j) 
       c = 0;
       x_j_norm = x_norm(j);
       
       if (x_j_norm == 0)
           r = 0;
       else
           
           gamma = stepsize * x_j_norm;
           a = z' * x_train(:, j);
           
           iter = 0;
           while (iter < 13)
               s = - y_train(j) / (1 + exp(y_train(j) * c));
               c = c - ( gamma * s + c - a ) ./ (gamma *exp(y_train(j)*c) * s *s + 1);
               iter = iter + 1;
           end
           r = (a - c) / (x_j_norm * stepsize);
       end
    end

%     Problem.v_dot_y = @v_dot_y;
%     function r = v_dot_y(z, stepsize, j)
%        c = 0;
%        a = z' * x_train(:, j);
%        gamma = stepsize* x_norm(j);
%        
%        iter = 0;
%        while (iter < 13)
%            s = - y_train(j) / (1 + exp(y_train(j) * c));
%            c = c - ( gamma * s + c -a ) / (gamma *exp(y_train(j)*c) * s *s + 1);
%            iter = iter + 1;
%        end
%        r = c / x_norm(j);
%     end


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%  for sdca        %%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Problem.init_v = @(alpha) sum(repmat(alpha, d, 1) .* x_train, 2) / (lambda2 * n_train);
    Problem.init_alpha = @() y_train/1000;

%     
   % delta alpha(in SDCA)
%     function r = conj_psi(b)
%         r = b * log(b) + (1-b) * log(1-b);
%     end
% 
%     Problem.delta_alpha = @delta_alpha;
%     function a = delta_alpha(alpha, w, i, kpa)
%         x = -y_train(i) * x_train(:, i);
%         lambda = lambda2 + kpa;
%         
%         p = x' * w;
%         q = -1 / (1 + exp(-p)) - alpha(i);
%         temp1 = log(1+exp(p)) + conj_psi(-alpha(i)) + p*alpha(i) + 2*q'*q;
%         temp2 = q'*q * (4 + (x' * x) / (lambda * n_train) );
%         s = min(1, temp1/temp2);
%         a = s*q;
%     end
    
% 
    Problem.delta_alpha = @delta_alpha;
    function a = delta_alpha(alpha, w, i, kpa)
        temp1 = (1 + exp(x_train(:, i)' * w * y_train(i)) )^(-1) * y_train(i) - alpha(i);
        temp2 = max(1, 0.25 + x_train(:, i)' * x_train(:, i)/((lambda2 + kpa)*n_train) );
        a = temp1 / temp2;

        lambda = (lambda2 + kpa) * n_train;
        w_x = w' * x_train(:, i);
        y_2 = y_train(i)'*y_train(i);
        
        iter = 0;
        while (iter < 20)
           a_dot_y = (alpha(i) + a) *y_train(i);
           a1 = y_train(i)*log(a_dot_y) - y_train(i)*log(1- a_dot_y) + w_x + (a*x_norm(i))/lambda;
           a2 = y_2/a_dot_y + y_2/(1-a_dot_y) + x_norm(i)/lambda;
           a = a -0.01* a1/a2;
           iter = iter + 1;
        end
    end

    Problem.delta_v = @delta_v;
    function r = delta_v(delta_alpha, i, kpa)
        r = delta_alpha * x_train(:, i) / ((lambda2 + kpa) * n_train);
    end
    
    % calculate proximal of loss f_j
    Problem.prox_loss = @prox_loss;
    % t: stepsize
    function s = prox_loss(z, j, t)
        mu = 1 - lambda2 * t / (1 + lambda2 * t);
        s = prox_f_j(mu * z, j, mu * t);
    end

    function r = prox_f_j(z, j, stepsize)
        % initial iteration
        c = 0;
        x_j_norm = x_norm(j);
        gamma = stepsize * x_j_norm;
        a = z' * x_train(:, j);
        
        iter = 0;
        while (iter < 13)
            s = - y_train(j) / (1 + exp(y_train(j) * c));
            c = c - ( gamma * s + c - a ) ./ ( 1 - y_train(j) * s - gamma * s *s);
            iter = iter + 1;
        end
         
        r = z - (a - c) * x_train(:, j) / x_j_norm;
    end

    % proximal of l1 
    Problem.prox = @l1_soft_thresh;
    function v = l1_soft_thresh(w, t)
       v = soft_thresh(w, t * lambda1);
    end    

    Problem.hess = @hess; 
    function h = hess(w, indices)
        sigm_val = sigmoid(y_train(indices).*(w'*x_train(:,indices)));
        c = sigm_val .* (ones(1,length(indices))-sigm_val); 
        h = 1/length(indices)* x_train(:,indices) * diag(y_train(indices).^2 .* c) * x_train(:,indices)'+lambda*eye(d);     
    end
    Problem.full_hess = @(w) hess(w, 1:n_train);
    
    % prediction results
    Problem.prediction = @prediction;
    function p = prediction(w)
        p = sigmoid(w' * x_test);
        class1_idx = p>0.5;
        class2_idx = p<=0.5;         
        p(class1_idx) = 1;
        p(class2_idx) = -1;          
    end

    % prediction accuracy
    Problem.accuracy = @(y_pred) sum(y_pred == y_test) / n_test;

    % calculate solution
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