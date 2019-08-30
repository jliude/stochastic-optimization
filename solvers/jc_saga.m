function [ w, infos ] = jc_saga(problem, in_options)
% Jingchang defined SAGA algorithm
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
   
    % define local_options
    local_options = [];
    
    % merge options
    options = merge_two_options(get_default_options(n, d), local_options);
    options = merge_two_options(options, in_options);

    % point_saga can not deal with mini-batch case
    if (options.batch_size ~= 1)
        disp('Error! batch_size != 1')
        return
    end
    
    % initialize
    total_iter = 0;
    epoch = 0;
    grad_calc_count = 0;
    w = options.w_init;
    num_of_bachces = floor(n / options.batch_size);
    stepsize = options.stepsize;
    
    % prepare an array of gradient mapping, and averge gradient
    grad_array = zeros(d, num_of_bachces);
    grad_ave = mean(grad_array, 2);
    
    % store first infos
    clear infos;
    [infos, f_val, optgap] = store_infos(problem, w, options, [], epoch, grad_calc_count, 0);
    
    % display infos
    if options.verbose == true
        fprintf('jc-saga: Epoch = %03d, cost = %.16f, optgap = %.4f\n', epoch, f_val, optgap);
    end
    
    % set start time
    start_time = tic();

    % main loop
    while (optgap > options.tol_optgap) && (epoch < options.max_epoch)
        
        for j = 1: num_of_bachces
            
            % select an index to update
            idx = randi(num_of_bachces);
            
            % update w
            z = problem.prox_loss(w, idx, stepsize);
            w_old = w;
            w = z + stepsize * (grad_array(:, idx) - grad_ave);
            
            % update grad_ave
            grad_mapping = (w_old - z) / stepsize;
            grad_ave = grad_ave + ( grad_mapping - grad_array(:, idx) ) / num_of_bachces;
            
            % replace with new gradient mapping
            grad_array(:, idx) = grad_mapping;
            
            % proximal operator, deal with nonsmooth regularization
            if isfield(problem, 'prox')
                w = problem.prox(w, stepsize);
            end
            
            total_iter = total_iter + 1;
        end
        % measure elasped time
        elasped_time = toc(start_time);
        
        % count gradient evaluations
        grad_calc_count = grad_calc_count + num_of_bachces;
        epoch = epoch + 1;
        
        % store infos
        [infos, f_val, optgap] = store_infos(problem, w, options, infos, epoch, grad_calc_count, elasped_time);
        w_gap = ( w - options.w_opt )' * (w - options.w_opt);

        % display infos
        if options.verbose > 0
            fprintf('jc-saga: PASS_GRAD = %d, cost = %.16f, optgap = %.8f, time = %.6f, w_gap = %.20f\n', grad_calc_count/num_of_bachces, f_val, optgap, elasped_time, w_gap);
        end
    end
    
    if optgap < options.tol_optgap
        fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', options.tol_optgap);
    elseif epoch == options.max_epoch
        fprintf('Max epoch reached: max_epochr = %g\n', options.max_epoch);
    end
end

