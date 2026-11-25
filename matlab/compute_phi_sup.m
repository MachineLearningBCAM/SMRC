function [phi, y_pred, prob] = compute_phi_sup(x, param, Y)
% COMPUTE_PHI_SUP  Feature mapping Phi_sup(x, y) in eq. (3.6) of the paper:
%   Phi_sup(x, y) = e_y (x) Psi_sup(x)  if x is confidently classified
%                   1   (x) Psi_sup(x)  otherwise
%
%   [phi, y_pred, prob] = COMPUTE_PHI_SUP(x, param, Y)
%
%   x      : d x n matrix of instances
%   param  : supervised algorithms learned by learn_sup.m
%   Y      : number of classes
%
%   phi    : cell array; phi{i} is the Y x m2 matrix with rows
%            Phi_sup(x_i, 1), ..., Phi_sup(x_i, Y)
%   y_pred : labels assigned by the supervised classifier h_sup
%   prob   : Y x n matrix of classification scores

n = size(x, 2);
I = eye(Y);
phi = cell(1, n);
y_pred = zeros(1, n);
prob = zeros(n, Y);

for i = 1:n
    [y_pred(i), prob(i, :)] = predict(param.model, x(:, i)');
    xx = feat_sup(x(:, i), param);
    confident = prob(i, y_pred(i)) > param.th;
    N = zeros(Y, Y * length(xx));
    for k = 1:Y
        if confident
            N(k, :) = kron(I(:, k), xx)';
        else
            N(k, :) = kron(ones(Y, 1), xx)';
        end
    end
    phi{i} = N;
end
prob = prob';
end
