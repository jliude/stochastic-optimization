function Problem = sparse_svm_plus(x_train, y_train, x_test, y_test, lambda1, lambda2)
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

    % loss
    Problem.cost = @cost;
    function f = cost(w)
       
        f_sum = sum( max(0.0, 1 - y_train .* (w' * x_train) ));
        f = f_sum / n_train + lambda2/2 * (w' * w) + lambda1 * norm(w, 1); 
        
    end

    % proximal of l1 
    Problem.prox = @l1_soft_thresh;
    function v = l1_soft_thresh(w, t)
       v = 1/ ( 1 + t * lambda2) * soft_thresh(w, lambda1 * t);
    end

    % calculate proximal of loss f_j
    Problem.prox_loss = @prox_loss;
    % t: stepsize
    function r = prox_loss(z, j, stepsize)
       s = ( 1 - y_train(j) * z' * x_train(:, j) ) / ( stepsize * x_train(:, j)' * x_train(:, j) ); 
       v = -1 .* ( s >= 1)  +  0 .* ( s <= 0 ) + -s .* ( s > 0 & s < 1 );
       r = z - stepsize * y_train(j) * v * x_train(:, j);
    end

    % stochastic gradient 1/n * sum_i^n f_i(w) + 1/2 * lambda2 * w^2
    Problem.grad = @grad;
    function g = grad(w, indices)
        
        alpha = w' * x_train(:,indices);
        flag = y_train(indices) .* alpha;
        flag(flag <= 1.0) = 1;
        flag(flag > 1.0) = 0;
        
        g = -sum(flag .* y_train(indices) .* x_train(:, indices), 2) / length(indices);
        
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

