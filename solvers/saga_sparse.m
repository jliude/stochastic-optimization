function [w, infos] = saga_sparse(problem, in_options)
% SAGA algorithm
% Deal to sparse problem in small memeory
% Inputs:
%       problem     function (cost/grad/hess)
%       in_options  options
% Output:
%       w           solution of w
%       infos       information

    % dimentions and samples
    d = problem.dim();
    n = problem.samples();
    
    % define local_options
    local_options = [];
    
    % merge options
    options = merge_two_options(get_default_options(d), local_options);
    options = merge_two_options(options, in_options);
    
    % initialize
    total_iter = 0;
    epoch = 0;
    grad_calc_count = 0;
    w = options.w_init;
    num_of_bachces = floor(n / options.batch_size);
    stepsize = options.stepsize;
    
    % prepare an array of gradients, and a valiable of average gradient
    grad_array = sparse(d, num_of_bachces);
    grad_ave = mean(grad_array, 2);
    
    % store first infos
    clear infos;
    [infos, f_val, optgap] = store_infos(problem, w, options, [], epoch, grad_calc_count, 0);
    
    % display infos
    if options.verbose == true
        fprintf('SAGA: Epoch = %03d, cost = %.16f, optgap = %.4f\n', epoch, f_val, optgap);
    end
    
    % set start time
    start_time = tic();
    
    % main loop
    while (optgap > options.tol_optgap) && (epoch < options.max_epoch)
        
        for j = 1 : num_of_bachces
            
            % selcet a min-batch to update 
            idx = randi(num_of_bachces);

            % get start_index, end_index
            start_index = (idx-1) * options.batch_size + 1;
            if idx < num_of_bachces
                end_index = start_index + options.batch_size - 1;
            else
                end_index = n;
            end

            % calculate gradient
            grad = problem.grad(w, start_index:end_index);
            
            % update w use saga gradient
            w = w - stepsize * ( grad_ave + grad - grad_array(:, idx));

            % update grad_ave
            grad_ave = grad_ave + (grad - grad_array(:, idx)) / num_of_bachces;
            
            % replace with new grad
            grad_array(:, idx) = grad;
            
            % proximal operator
            if isfield(problem, 'prox')
                w = problem.prox(w, stepsize);
            end
            
            total_iter = total_iter + 1;
        end
        
        % measure elasped time
        elapsed_time = toc(start_time);
        
        % count gradient evaluations
        grad_calc_count = grad_calc_count + num_of_bachces;
        epoch = epoch + 1;

        % store infos
        [infos, f_val, optgap] = store_infos(problem, w, options, infos, epoch, grad_calc_count, elapsed_time);           
      
        % display infos
        if options.verbose > 0
            fprintf('SAGA: PASS_GRAD = %d, cost = %.16f, optgap = %.10f, time = %.6f\n', grad_calc_count/num_of_bachces, f_val, optgap, elapsed_time);
        end
    end
    
    if optgap < options.tol_optgap
        fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', options.tol_optgap);
    elseif epoch == options.max_epoch
        fprintf('Max epoch reached: max_epochr = %g\n', options.max_epoch);
    end
end

