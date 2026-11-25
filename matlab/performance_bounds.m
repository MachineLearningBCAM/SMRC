function [lower, upper] = performance_bounds(tau, lambda, phi_x, h_x, iter)
% PERFORMANCE_BOUNDS  High-confidence lower and upper bounds for the error
% probability of a classification rule h (Theorem 6 in the paper).
%
%   [lower, upper] = PERFORMANCE_BOUNDS(tau, lambda, phi_x, h_x, iter)
%
%   tau, lambda : mean and confidence vectors such that
%                 P{|E{Phi} - tau| < lambda} >= 1 - delta
%   phi_x       : cell array; phi_x{i} is the Y x m matrix with rows
%                 Phi(x_i, 1), ..., Phi(x_i, Y) for unlabeled instances
%   h_x         : Y x n matrix with entries h(y|x_i)
%   iter        : number of iterations of the stochastic subgradient method
%
%   With probability at least 1 - delta, lower <= R(h) <= upper, where
%   upper = min_mu 1 - tau'mu + lambda'|mu| + E{max_y Phi(x,y)'mu - h(y|x)}
%   lower = max_mu 1 - tau'mu - lambda'|mu| + E{min_y Phi(x,y)'mu - h(y|x)}

m = length(tau);
n = length(phi_x);
th = 10^-6;
step = 0.1 / m;

muu_lower = sparse(zeros(m, 1));
muu_upper = sparse(zeros(m, 1));
muu_aux_lower = zeros(m, 1);
muu_aux_upper = zeros(m, 1);
muu_ave_lower = zeros(m, 1);
muu_ave_upper = zeros(m, 1);

cont = 1;
while cont < iter
    for i = 1:n
        v_lower = phi_x{i} * muu_lower - h_x(:, i);
        v_upper = phi_x{i} * muu_upper - h_x(:, i);
        [~, y_lower] = min(v_lower);
        [~, y_upper] = max(v_upper);

        grad_lower = -phi_x{i}(y_lower, :)';
        grad_upper = phi_x{i}(y_upper, :)';

        muu_aux_lower = muu_aux_lower - step * (tau + lambda .* sign(muu_aux_lower) + grad_lower);
        muu_aux_upper = muu_aux_upper - step * (-tau + lambda .* sign(muu_aux_upper) + grad_upper);

        idx = find(abs(muu_aux_lower) > th);
        muu_lower = sparse(idx, ones(1, length(idx)), muu_aux_lower(idx), m, 1);
        idx = find(abs(muu_aux_upper) > th);
        muu_upper = sparse(idx, ones(1, length(idx)), muu_aux_upper(idx), m, 1);

        cont = cont + 1;
        muu_ave_lower = ((cont - 1) / cont) * muu_ave_lower + (1 / cont) * muu_aux_lower;
        muu_ave_upper = ((cont - 1) / cont) * muu_ave_upper + (1 / cont) * muu_aux_upper;
    end
end

value_lower = 0;
value_upper = 0;
for i = 1:n
    value_lower = value_lower + min(phi_x{i} * muu_ave_lower - h_x(:, i)) / n;
    value_upper = value_upper + max(phi_x{i} * muu_ave_upper - h_x(:, i)) / n;
end

lower = 1 - tau' * muu_ave_lower - lambda' * abs(muu_ave_lower) + value_lower;
upper = 1 - tau' * muu_ave_upper + lambda' * abs(muu_ave_upper) + value_upper;
end
