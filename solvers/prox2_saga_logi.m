function [ w, infos ] = prox2_saga_logi(problem, in_options)
% low storage prox2-saga
%
% Iuput:
%       problem:    function(cost/grad/etc)
%       in_options  options
% Output:
%       w           soulutions of w
%       infos       information

    % dimension and samples
    d = problem.dim();
    n = problem.samples();

    % set local.options
    local_options.sub_mode = [];

    % merge options
    options = merge_two_options(get_default_options(n, d), local_options);
    options = merge_two_options(options, in_options);

    % intialize
    total_iter = 0;
    epoch = 0;
    grad_calc_count = 0;
    w = options.w_init;
    num_of_bachces = floor(n / options.batch_size);
    stepsize = options.stepsize;

    % training data
    lambda2 = problem.lambda2();
    lambda1 = problem.lambda1();

    % prepare an array of v_dot_y, and average gradeint
    grad_array = zeros(num_of_bachces, 1);
    grad_ave = 0;

    % store first infos
    clear infos;
    [infos, f_val, optgap] = store_infos(problem, w, options, [], epoch, grad_calc_count, 0);

    % display infos
    if options.verbose == true
        fprintf('Point-saga: Epoch = %03d, cost = %.16f, optgap = %.4f\n', epoch, f_val, optgap);
    end

    % set start time
    start_time = tic();

    % main loop
    while (optgap > options.tol_optgap) && (epoch < options.max_epoch)

        for j = 1: 10000

            % select a mini-batch to update
            idx = randi(num_of_bachces);

            % update w
            z = w + stepsize * (grad_array(idx) .* problem.x_train_i(idx) - grad_ave );
            
            v_dot_y = problem.v_dot_y(z, stepsize, idx);
            y = v_dot_y * problem.x_train_i(idx);
            grad_mapping = (z - y) / stepsize;

            % update grad_ave
            grad2 = ( z - grad_array(idx) * problem.x_train_i(idx) ) /stepsize;
            grad_ave = grad_ave + ( grad_mapping - grad2 ) / num_of_bachces;

            % update grad_array
            grad_array(idx) = v_dot_y;

            % proximal operator, deal with nonsmooth regularization
            if isfield(problem, 'prox')
                mu = 1 / (1 + lambda2 * stepsize);
                w = mu * problem.prox(y, stepsize);
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
        w_gap = ( w - options.w_opt )' * (w - options.w_opt);

        % display infos
        if options.verbose > 0
            fprintf('Prox2-SAGA: PASS_GRAD = %d, cost = %.16f, optgap = %.10f, time = %.6f, w_gap = %.20f\n', grad_calc_count/num_of_bachces, f_val, optgap, elapsed_time, w_gap);
        end
    end

    if optgap < options.tol_optgap
        fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', options.tol_optgap);
    elseif epoch == options.max_epoch
        fprintf('Max epoch reached: max_epochr = %g\n', options.max_epoch);
    end
end
