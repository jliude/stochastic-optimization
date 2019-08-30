function Problem = sparse_svm(x_train, y_train, x_test, y_test, lambda1, lambda2)
% This file defines l2-regularized SVM problem
%
% Inputs:
%       x_train     train data matrix of x of size dxn.
%       y_train     train data vector of y of size 1xn.
%       x_test      test data matrix of x of size dxn.
%       y_test      test data vector of y of size 1xn.
%       lambda2     l2-regularized parameter. 
%       lambda1     l1-regularized parameter.
% Output:
%       Problem     problem instance. 
%
%
% The problem of interest is defined as
%
%           min f(w) = 1/n * sum_i^n f_i(w) + 1/2 * lambda2 * w^2 + lambda1 * || w ||_1,           
%           where 
%           f_i(w) = max(0.0, 1 - y_i .* (w'*x_i) ) .
%
% "w" is the model parameter of size d vector.
%

    % sample and dimension size
    d = size(x_train, 1);
    n_train = length(y_train);
    n_test = length(y_test);

    Problem.name = @() 'sparse svm';
    Problem.dim = @() d;
    Problem.samples = @() n_train;
    Problem.classes = @() 2;
    Problem.hessain_w_independent = @() false;
    Problem.lambda1 = @() lambda1;
    Problem.lambda2 = @() lambda2;
    Problem.x_train = @() x_train;
    Problem.y_train = @() y_train;

    % loss
    Problem.cost = @cost;
    function f = cost(w)
       
        f_sum = sum( max(0.0, 1 - y_train .* (w' * x_train) ));
        f = f_sum / n_train + lambda2/2 * (w' * w) + lambda1 * norm(w, 1); 
        
    end

    % proximal of l1 
    Problem.prox = @l1_soft_thresh;
    function v = l1_soft_thresh(w, t)
       v = soft_thresh(w, t * lambda1);
    end

    % calculate proximal of loss f_j
    Problem.prox_loss = @prox_loss;
    % t: stepsize
    function s = prox_loss(z, j, t)
        mu = 1 / (1 + lambda2 * t);
        s = prox_f_j(mu * z, j, mu * t);
    end

    function r = prox_f_j(z, j, stepsize)
       s = ( 1 - y_train(j) * z' * x_train(:, j) ) / ( stepsize * x_train(:, j)' * x_train(:, j) ); 
       v = -1 .* ( s >= 1)  +  0 .* ( s <= 0 ) + -s .* ( s > 0 & s < 1 );
       r = z - stepsize * y_train(j) * v * x_train(:, j);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%% low storage prox2-saga %%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Problem.v_dot_y = @v_dot_y;
    function r = v_dot_y(z, stepsize, j)
        s = ( 1 - y_train(j) * z' * x_train(:, j) ) / ( stepsize * x_train(:, j)' * x_train(:, j) );
        v = -1 .* ( s >= 1)  +  0 .* ( s <= 0 ) + -s .* ( s > 0 & s < 1 );
        r = y_train(j) * v;
    end

    Problem.x_train_i = @(i) x_train(:, i);

    % delta alpha(in SDCA)
    Problem.delta_alpha = @delta_alpha;
    function a = delta_alpha(alpha, w, i, kpa)
        % alpha: dual variable  w: primal variable 
        % i: index  lambda: regular term
        x_i_norm = x_train(:, i)' * x_train(:, i);
        temp = (1 - x_train(:, i)' * w * y_train(i) )/( x_i_norm / ((lambda2 + kpa) * n_train)) + alpha(i) * y_train(i);
        a = y_train(i) * max(0, min(1, temp)) - alpha(i);
        
    end

    Problem.delta_v = @delta_v;
    function r = delta_v(delta_alpha, i, kpa)
        r = delta_alpha * x_train(:, i) / ((lambda2 + kpa) * n_train);
    end

    % stochastic gradient 1/n * sum_i^n f_i(w) + 1/2 * lambda2 * w^2
    Problem.grad = @grad;
    function g = grad(w, indices)
        
        alpha = w' * x_train(:,indices);
        flag = y_train(indices) .* alpha;
        flag(flag <= 1.0) = 1;
        flag(flag > 1.0) = 0;
        
        g = lambda2 * w - sum(flag .* y_train(indices) .* x_train(:, indices), 2) / length(indices);
        
    end

    Problem.stored_var = @stored_var;
    function r = stored_var(w, i)
        
        alpha = w' * x_train(:,i);
        flag = y_train(i) .* alpha;
        flag(flag <= 1.0) = 1;
        flag(flag > 1.0) = 0;
        
        r = - flag .* y_train(i);
    end

    % full gradient
    Problem.full_grad = @full_grad;
    function g = full_grad(w)
       
        g = grad(w, 1:n_train);
        
    end

    Problem.prediction = @prediction;
    function p = prediction(w)
        
        p = w' * x_test;
        
        class1_idx = p>0;
        class2_idx = p<=0;         
        p(class1_idx) = 1;
        p(class2_idx) = -1;        
        
    end

    Problem.accuracy = @accuracy;
    function a = accuracy(y_pred)
        
        a = sum(y_pred == y_test) / n_test; 
        
    end
% 
%     Problem.calc_solution = @calc_solution;
%     function w_opt = calc_solution(problem, maxiter)
%        
%         options.max_iter = maxiter;
%         options.verbose = true;
%         options.tol_optgap = 1.0e-10;
%         options.tol_gnorm = 1.0e-6;
%         options.w_opt = inf;
%         
%         [w_opt, ~] = point_saga(problem, options);
%         
%     end
    Problem.calc_solution = @calc_solution;
    function w_opt = calc_solution(problem, maxiter)
        
        options.max_iter = maxiter;
        options.verbose = true;
        options.tol_optgap = 1.0e-24;        
        options.tol_gnorm = 1.0e-16;
        options.step_alg = 'backtracking';        
        [w_opt,~] = gd(problem, options);
        
    end
end

