function [ w, infos ] = sgd( problem, in_options )
% Stochastic gradient descent (SGD) algorithm.
%
% Inputs:
%       problem     function (cost/grad/hess)
%       in_options  options
% Output:
%       w           solution of w
%       infos       information

    % dimensions and samples
    d = problem.dim();
    n = problem.samples();
    
    % set local options
    local_options = struct('stepsize', 0.01);
    
    % merge options
    options = merge_two_options(get_default_options(n, d), local_options);
    options = merge_two_options(options, in_options);
    
    % initialize
    total_iter = 0;
    epoch = 0;
    grad_cal_count = 0;
    w = options.w_init;
    num_of_bachces = floor(n / options.batch_size);
    
    % store first infos
    clear infos;
    [ infos, f_vals, optgap ] = store_infos(problem, w, options, [], epoch, grad_cal_count, 0);
    
    % display infos
    if options.verbose > 0
       fprintf('SGD: Epoch = %03d, cost = %.16e, optgap = %.4e\n', epoch, f_vals, optgap); 
    end
    
    % start time
    start_time = tic();
    
    % main loop
    while (optgap > options.tol_optgap) && (epoch < options.max_epoch)
        
        for j = 1 : 4000
            
            % selcet a min-batch to update 
            idx = randi(num_of_bachces);
            stepsize = options.stepsizefun(total_iter, options);
            % stepsize = 0.001;
            
            % get start_index, end_index
            start_index = (idx-1) * options.batch_size + 1;
            if idx < num_of_bachces
                end_index = start_index + options.batch_size - 1;
            else
                end_index = n;
            end

            % calculate gradient
            grad = problem.grad(w, start_index:end_index);
            
            % update w
            w = w - stepsize * grad;
            
            total_iter = total_iter + 1;
    
        end
        
        % measure elasped time
        elasped_time = toc(start_time);
        
        % count gradient evaluations
        grad_cal_count = grad_cal_count + num_of_bachces;
        epoch = epoch + 1;
        
        % store infos
        [infos, f_val, optgap] = store_infos(problem, w, options, infos, epoch, grad_cal_count, elasped_time);
%         w_gap = ( w - options.w_opt )' * (w - options.w_opt);

        % display infos
        if options.verbose > 0
            fprintf('SGD: PASS_GRAD = %d, cost = %.16f, optgap = %.8f, time = %.6f\n', grad_cal_count/num_of_bachces, f_val, optgap, elasped_time);
        end
    end
    
    if optgap < options.tol_optgap
        fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', options.tol_optgap);
    elseif epoch == options.max_epoch
        fprintf('Max epoch reached: max_epochr = %g\n', options.max_epoch);
    end
end

