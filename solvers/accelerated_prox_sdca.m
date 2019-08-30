function [ w, infos ] = accelerated_prox_sdca(problem, in_options)
% accelerated proximal sdca
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
    lambda1 = problem.lambda1();
    
    local_options = [];
    
    % merge options
    options = merge_two_options(get_default_options(n, d), local_options);
    options = merge_two_options(options, in_options);
    
    % initialize
    total_iter = 0;
    epoch = 0;
    grad_cal_count = 0;
    w = options.w_init;
    y = w;
    v = w;
    alpha = options.alpha_init;
    num_of_bachces = floor(n / options.batch_size);

    % parameters in accelerated proximal sdca
    kpa = 0.0001 * lambda2;
    mu = lambda2 / 2;
    rho = mu + kpa;
    eta = sqrt(mu / rho);
    beta = ( 1 - eta ) / ( 1 + eta );
    
    % store first infos
    clear infos;
    [infos, f_val, optgap] = store_infos(problem, w, options, [], epoch, grad_cal_count, 0);
    
    % display infos
    if options.verbose == true
        fprintf('Point-saga: Epoch = %03d, cost = %.16f, optgap = %.4f\n', epoch, f_val, optgap);
    end
    
    % set start time
    start_time = tic();
    
    % main loop
    while (optgap > options.tol_optgap) && (epoch < options.max_epoch)
       
        for j = 1 : 2000
            
            % select a mini-batch to update
            idx = randi(num_of_bachces);
            
            % batch_size = 1
            delta_alpha = problem.delta_alpha(alpha, w, idx, kpa);
            
            % update alpha
            alpha(idx) = alpha(idx) + delta_alpha;
            
            % update w
            w_old = w;
            v = v + problem.delta_v(delta_alpha, idx, kpa);
            
            % proximal operator
            if isfield(problem, 'prox')
               %w = problem.prox((lambda1 * v + kpa * y) / (lambda2 + kpa) , (lambda2 + kpa) ); 
               w = problem.prox(v + y * kpa/(lambda2 + kpa), 1/(lambda2 + kpa)); 
            end
            
            % update y
            y = w + beta * (w - w_old);
            
            total_iter = total_iter + 1;
        end
        
        % measure elapsed time
        elapsed_time = toc(start_time);
        
        % count gradient evaluations
        grad_cal_count = grad_cal_count + num_of_bachces;
        epoch = epoch + 1;
        
        % store infos
        [infos, f_val, optgap] = store_infos(problem, w, options, infos, epoch, grad_cal_count, elapsed_time);
        w_gap = ( w - options.w_opt )' * (w - options.w_opt);
        
        % display infos
        if options.verbose > 0
            fprintf('Accelerated SDCA: PASS_GRAD = %d, cost = %.16f, optgap = %.8f, time = %.6f, w_opt = %.20f\n', grad_cal_count/num_of_bachces, f_val, optgap, elapsed_time, w_gap);
        end
    end
end

