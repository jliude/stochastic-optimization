function [ w, infos ] = point_saga_plus(problem, in_options)
%  More efficirnt point stochastic average descent (Point-SAGA) algorithm
%
% Inputs:
%       problem     function (cost/grad/etc)
%       in_options  options
% Output:
%       w           solutions of w
%       infos       information
    
    % dimensions and samples
    d = problem.dim();
    n = problem.samples();
    local_options = [];
    
    % merge options
    options = merge_two_options(get_default_options(d), local_options);
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
    
    % prepare an array of gradient mapping, and average gradient
    grad_array = zeros(d, num_of_bachces);
    grad_ave = mean(grad_array, 2);
    
    % store first infos
    clear infos;
    [infos, f_val, optgap] = store_infos(problem, w, options, [], epoch, grad_calc_count, 0);
        
    % display infos
    if options.verbose == true
        fprintf('Point-saga-plus: Epoch = %03d, cost = %.16f, optgap = %.4f\n', epoch, f_val, optgap);
    end
    
    % set start time
    start_time = tic();
    
    % main loop
    while (optgap > options.tol_optgap) && (epoch < options.max_epoch)
       
        for j = 1:num_of_bachces
            
            % select a min-batch to update
            idx = randi(num_of_bachces);
            
            % dicide what algorithm to use
            if(abs(min(w)) <= options.tolerate + options.nonsmooth_point)
               
               % point-saga, update w
               z = w + stepsize * (grad_array(:, idx) - grad_ave);
               w = problem.prox_loss(z, idx, stepsize);
               grad = (z - w) / stepsize;
               
            else
               
                % saga, update w
                grad = problem.grad(w, idx);
                w = w - stepsize * ( grad_ave + grad - grad_array(:, idx));

            end
            
            % update grad_ave
            grad_ave = grad_ave + (grad - grad_array(:, idx)) / num_of_bachces;
            
            % replace eith new grad
            grad_array(:, idx) = grad;

            
            % proximal operator, deal with nonsmooth regularization
            if isfield(problem, 'prox')
                w = problem.prox(w, stepsize);
            end
             
            total_iter = total_iter + 1;            
        end
        
        % measure elasped time
        elapsed_time = toc(start_time);
        
        % count gradient evalution
        grad_calc_count = grad_calc_count + num_of_bachces;
        epoch = epoch + 1;
        
        % store infos
        [infos, f_val, optgap] = store_infos(problem, w, options, infos, epoch, grad_calc_count, elapsed_time);           
            
        % display infos
        if options.verbose > 0
            fprintf('Point-SAGA-plus: PASS_GRAD = %d, cost = %.16f, optgap = %.10f, time = %.6f\n', grad_calc_count/num_of_bachces, f_val, optgap, elapsed_time);
        end
    end
    
    if optgap < options.tol_optgap
        fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', options.tol_optgap);
    elseif epoch == options.max_epoch
        fprintf('Max epoch reached: max_epochr = %g\n', options.max_epoch);
    end    
end

