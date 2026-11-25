function [muu_sol, minimax] = fit_SMRC(tau, lambda, phi_x, Y, iter, B)
% FIT_SMRC  Learn SMRC parameters by solving problem (3.2) in the paper.
%
%   [muu_sol, minimax] = FIT_SMRC(tau, lambda, phi_x, Y, iter, B)
%
%   tau, lambda : mean and confidence vectors (m x 1)
%   phi_x       : cell array; phi_x{i} is the Y x m matrix with rows
%                 Phi(x_i, 1), ..., Phi(x_i, Y) for the instances used to
%                 compute stochastic subgradients
%   Y           : number of classes
%   iter        : number of iterations of the stochastic subgradient method
%   B           : number of instances per stochastic subgradient
%
%   muu_sol     : SMRC parameters (averaged iterates)
%   minimax     : minimax risk R(V) (objective value at muu_sol)

m = length(tau);
n = length(phi_x);

muu = sparse(zeros(m, 1));
muu_aux = zeros(m, 1);
muu_sol = zeros(m, 1);
th = 10^-6;        % components of muu below th are set to zero
step = 0.1 / m;    % step size

cont = 1;
while cont < iter
    for i = 1:B:n - B + 1
        grad = zeros(m, 1);
        for j = 0:B - 1
            grad = grad + (1 / B) * subgradient(phi_x{i + j}, muu, Y);
        end
        muu_aux = muu_aux - step * (-tau + lambda .* sign(muu_aux) + grad);

        idx = find(abs(muu_aux) > th);
        muu = sparse(idx, ones(1, length(idx)), muu_aux(idx), m, 1);

        cont = cont + 1;
        muu_sol = ((cont - 1) / cont) * muu_sol + (1 / cont) * muu_aux;
    end
end

value = 0;
for i = 1:n
    [~, varphi] = subgradient(phi_x{i}, muu_sol, Y);
    value = value + varphi / n;
end

minimax = 1 - tau' * muu_sol + lambda' * abs(muu_sol) + value;
end
