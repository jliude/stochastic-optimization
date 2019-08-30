function [ w, infos ] = Katyusha(problem, in_options)
% KATYUSHA acceleration
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
    % lambda1 = problem.lambda1();
    lipschitz = 0.5;
    
    local_options = [];
    
    % merge options
    options = merge_two_options(get_default_options(n, d), local_options);
    options = merge_two_options(options, in_options);
    
    % initialize
    total_iter = 0;
    epoch = 0;
    grad_calc_count = 0;
    w = options.w_init;
    
    % parameters in Katyusha
    m = 2 * n; % number of inner loop
    tau2 = 1/2; 
    tau1 = min(sqrt(m*lambda2)/sqrt(3*lipschitz), 1/2);
    alpha = 1/(3*tau1*lipschitz);
    w_tilde = w;
    z = w;
    y = w;
    radix = 1 + alpha*lambda2;
    
    % store first infos
    clear infos;
    [infos, f_val, optgap] = store_infos(problem, w, options, [], epoch, grad_calc_count, 0);
    
    % display infos
     if options.verbose == true
        fprintf('Katyusha: Epoch = %03d, cost = %.16f, optgap = %.4f\n', epoch, f_val, optgap);
    end
    
    % set start time
    start_time = tic();
    
    % main loop
    while (optgap > options.tol_optgap) && (epoch < options.max_epoch)
        
        % compute full gradient
        full_grad = problem.full_grad(w_tilde);
        
        temp = 0;
        grad_calc_count = grad_calc_count + n;
        epoch = epoch +1;
        
        for j = 0 : (m-1)
            
            % select an index to update
            idx = randi(n);
            
            w = tau1 * z + tau2 * w_tilde + (1-tau1-tau2) * y;
            grad = problem.grad(w, idx) - problem.grad(w_tilde, idx) + full_grad;
            z_old = z;
            
            z = z - alpha * grad;
            % proximal operator for l1
            if isfield(problem, 'prox')
                z = problem.prox(z, alpha);
            end
            
            % acceleration
            y = w + tau1 * (z - z_old); 
            
            total_iter = total_iter + 1;
            
            % store information
            grad_calc_count = grad_calc_count + 2;
            if mod(j, 1000) == 0
                elapsed_time = toc(start_time);
                [infos, f_val, ~] = store_infos(problem, w, options, infos, epoch, grad_calc_count, elapsed_time);
            end
            
            % cache
            temp = temp + radix^j * y;
        end
        
        % update w_tilde
        j = 0:m - 1;
        w_tilde = sum(radix.^j)^(-1) * temp;
        
        % count gradient evalutions
        epoch = epoch + 2;

        % display infos
        if options.verbose > 0
            fprintf('Katyusha: epoch = %03d, cost = %.16f, optgap = %.10f, time = %.6f\n', epoch, f_val, optgap, elapsed_time);
        end
    end
    
    % print results
    if optgap < options.tol_optgap
        fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', options.tol_optgap);
    elseif epoch == options.max_epoch
        fprintf('Max epoch reached: max_epochr = %g\n', options.max_epoch);
    end
end

