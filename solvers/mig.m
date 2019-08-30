function [ w, infos ] = mig(problem, in_options)
% Mig accelerate
% 
% Inputs: 
%       problem     function (cost/grad/etc)
%       in_options  options
% Output:
%       w       solutions of w
%       infos   information

    % dimensions and samples
    d = problem.dim();
    n = problem.samples();
    lambda2 = problem.lambda2();
    local_options = [];
    
    % merge options
    options = merge_two_options(get_default_options(n, d), local_options);
    options = merge_two_options(options, in_options);
    
    % initialize
    total_iter = 0;
    epoch = 0;
    grad_calc_count = 0;
    w = options.w_init;
    w_tilde = w;
    stepsize = options.stepsize;
    radix = 1 + stepsize*lambda2;
    
    % parameters in Mig
    m = 2 * n; % number of inner loop
    theta = 0.4;    
        
    % store first infos
    clear infos;
    [infos, f_val, optgap] = store_infos(problem, w, options, [], epoch, grad_calc_count, 0);
    
    % display infos
     if options.verbose == true
        fprintf('Mig: Epoch = %03d, cost = %.16f, optgap = %.4f\n', epoch, f_val, optgap);
    end
    
    % set start time
    start_time = tic();
    
    % main loop
    while (optgap > options.tol_optgap) && (epoch < options.max_epoch)
        
        % complete full gradient
        full_grad = problem.full_grad(w_tilde);
        
        temp = 0;
        grad_calc_count = grad_calc_count + n;
        
        for j = 0 : (m-1)
           
            % select an index to update
            idx = randi(n);
            
            y = theta * w + (1 - theta) * w_tilde;
            grad = problem.grad(y, idx) - problem.grad(w_tilde, idx) + full_grad;
            
            % update
            w = w - stepsize * grad;
            
            % proximal operator, deal with nonsmooth regularization
            if isfield(problem, 'prox')
                mu = 1 / (1 + lambda2 * stepsize);
                w = mu * problem.prox(w, stepsize);
            end
            
            % record
            grad_calc_count = grad_calc_count + 2;
            
            % measure elapsed time
            elapsed_time = toc(start_time);
            if(mod(grad_calc_count, 500) == 0)
                % store infos
                [infos, f_val, optgap] = store_infos(problem, w, options, infos, epoch, grad_calc_count, elapsed_time);
            end
            
            % cache
            temp = temp + radix^j * y;
        end
        
        % update w_tilde
        i = 0:(m-1);
        w_tilde = sum(radix.^i)^(-1) * temp;
        
        % count gradient evalutions
        epoch = epoch + 1;

        % display infos
        if options.verbose > 0
            fprintf('Mig: epoch = %03d, cost = %.16f, optgap = %.10f, time = %.6f\n', epoch, f_val, optgap, elapsed_time);
        end
    end
    
    % print results
    if optgap < options.tol_optgap
        fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', options.tol_optgap);
    elseif epoch == options.max_epoch
        fprintf('Max epoch reached: max_epochr = %g\n', options.max_epoch);
    end
end

